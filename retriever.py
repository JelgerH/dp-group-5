import numpy as np
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v1.5"
FULL_DIM = 768 # native dimensionality of the model
NUM_DOCS = 1000
NUM_QUERIES = 200
TOP_K = 10

device = torch.device("cpu")


def load_corpus_and_queries():
    """Build the doc DB (unique contexts) and query set (questions)."""
    ds = load_dataset("rajpurkar/squad", split="validation")

    # Unique contexts, preserving first-seen order
    contexts, seen = [], set()
    for ex in ds:
        c = ex["context"]
        if c not in seen:
            seen.add(c)
            contexts.append(c)
        if len(contexts) == NUM_DOCS:
            break
    assert len(contexts) == NUM_DOCS

    # Queries: questions whose gold context is in the DB
    ctx_to_id = {c: i for i, c in enumerate(contexts)}
    queries, gold = [], []
    for ex in ds:
        c = ex["context"]
        if c in ctx_to_id:
            queries.append(ex["question"])
            gold.append(ctx_to_id[c])
        if len(queries) == NUM_QUERIES:
            break
    assert len(queries) == NUM_QUERIES
    return contexts, queries, gold


def encode(texts, is_query=False, batch_size=64):
    """Return one embedding per text, L2-normalized, on CPU.

    Arctic models are *asymmetric*: queries get a prompt prefix,
    documents are encoded as-is (this matches the model card).
    """
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    if is_query:
        texts = ["Represent this sentence for searching relevant passages: " + t
                 for t in texts]

    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(
                texts[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out = model(**batch)

            # mean pooling over non-padding tokens
            mask = batch["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb = F.normalize(emb, p=2, dim=1)
            vecs.append(emb)
    return torch.cat(vecs)  # (n, 768), float32, unit norm


def truncate(matryoshka_vecs, dim):
    """Keep the first `dim` components and re-normalize.

    Re-normalization is essential: the prefix has norm < 1 in general,
    and cosine similarity is scale-invariant, so this preserves the
    ranking geometry the model was trained to have.
    """
    return F.normalize(matryoshka_vecs[:, :dim], p=2, dim=1)

# --------------------------------
# Part (c): 1. cosine similarity function
#           2. top_k selection
# --------------------------------

def cosine_similarity(query_vec, doc_matrix):
    """Cosine similarity of one query against all documents.

    Derived from the Euclidian dot product formula:
    A @ B = norm(A) * norm(B) * cos(theta)
    <=>
    cos(theta) = (A @ B) / (norm(A) * norm(B))
    """

    q = np.asarray(query_vec, dtype=np.float64)
    D = np.asarray(doc_matrix, dtype=np.float64)
    dots = D @ q
    qn = np.linalg.norm(q)
    dn = np.linalg.norm(D, axis=1)
    return dots / np.maximum(qn * dn, 1e-12)

def top_k_indices(scores, k):
    """Indices of k largest scores, in descending order"""
    s = np.array(scores, dtype=np.float64, copy=True)
    k = min(k, s.shape[0])
    out = []
    for _ in range(k):
        i = int(np.argmax(s))
        out.append(i)
        s[i] = -np.inf
    return out

def recall_at_k(docs, queries, gold, k=TOP_K):
    """Recall@k: fraction of queries whose gold doc is in the top-k."""
    docs = np.asarray(docs)
    queries = np.asarray(queries)
    hits = 0
    for qi in range(queries.shape[0]):
        sims = cosine_similarity(queries[qi], docs)
        if gold[qi] in top_k_indices(sims, k):
            hits += 1
    return hits / queries.shape[0]

class RetrieverServer:
    """Class that stores the 1,000 documents and their vectors. The client
    may only talk to it through a single search(query_vector, k) call.
    """

    def __init__(self, contexts, embed_dim=256):
        # Embedding the corpus happens here, once.
        self.contexts = list(contexts)
        self.embed_dim = embed_dim
        D_full = encode(self.contexts, is_query=False)     # (1000, 768)

        # Matryoshka truncation to the working dimension, renormalized.
        self.doc_vecs = truncate(D_full, embed_dim)         # (1000, embed_dim)

    def search(self, query_vector, k):
        """Return the ids (and texts) of the k nearest documents.

        query_vector: (embed_dim,) numpy array or float tensor,
        assumed L2-normalized (the client normalizes before sending).
        """

        q = (query_vector.numpy() if isinstance(query_vector, torch.Tensor)
             else np.asarray(query_vector, dtype=np.float64))

        assert q.shape == (self.doc_vecs.shape[1],), \
            "query dimensionality must match the server's embed_dim"

        sims = cosine_similarity(q, self.doc_vecs)
        top_ids = top_k_indices(sims, k)

        return [
            {"id": i, "text": self.contexts[i],
             "score": float(sims[i])}
            for i in top_ids
        ]

def evaluate(server, Q, gold, ks=(1, 5, 10), warmup=10):
    """Recall@{1,5,10} and average per-query latency over the query set."""

    # ---- warmup (untimed) ----
    for q in Q[:warmup]:
        server.search(q, max(ks))

    # ---- timed pass ----
    hits = {k: 0 for k in ks}
    latencies = []
    for qi, qvec in enumerate(Q):
        t0 = time.perf_counter()
        results = server.search(qvec, max(ks)) # single round trip
        latencies.append(time.perf_counter() - t0)

        ids = [r["id"] for r in results]
        for k in ks:
            if gold[qi] in ids[:k]:
                hits[k] += 1

    n = len(Q)
    return ({k: hits[k] / n for k in ks},
            sum(latencies) / n,
            latencies)

def main():
    contexts, queries, gold = load_corpus_and_queries()

    # ---- Part (b): Client/Server split
    server = RetrieverServer(contexts, embed_dim=256)
    
    # ---- Client: encode queries, ask the server, measure Recall@10 --
    Q_full = encode(queries, is_query=True)     # (200, 768)
    Q = truncate(Q_full, 256)                   # (200, 256)

    # ---- Part (c): Evaluation
    recalls, avg_lat, lat = evaluate(server, Q, gold, ks=(1, 5, 10))

    print(f"\nEvaluation over {len(Q)} queries @ 256 dims, "
          f"{NUM_DOCS} docs")
    for k in (1, 5, 10):
        print(f"Recall@{k:<2} = {recalls[k]:.3f}")
    print(f"Avg query latency (server round trip): "
          f"{avg_lat * 1000:.2f} ms  (median "
          f"{sorted(lat)[len(lat) // 2] * 1000:.2f} ms)")

    # ---- Part (a): Matryoshka dimensions --------
    D_full = encode(contexts, is_query=False)   # (1000, 768)
    print(f"{'dim':>5} {'Recall@10':>10}")
    for d in [64, 128, 256, 768]:
        D, Q = truncate(D_full, d), truncate(Q_full, d)
        r = recall_at_k(D, Q, gold, TOP_K)
        print(f"{d:>5} {r:>10.3f}")

if __name__ == "__main__":
    main()
