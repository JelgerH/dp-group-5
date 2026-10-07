import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v1.5"
FULL_DIM = 768          # native dimensionality of the model
TRUNC_DIMS = [64, 128, 256, 768]
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


def recall_at_k(docs, queries, gold, k=TOP_K):
    """Recall@k: fraction of queries whose gold doc is in the top-k."""
    sims = queries @ docs.T                    # cosine, since normalized
    topk = sims.topk(k, dim=1).indices         # (n_q, k)
    hits = (topk == torch.tensor(gold).unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


def main():
    contexts, queries, gold = load_corpus_and_queries()

    D_full = encode(contexts, is_query=False)   # (1000, 768)
    Q_full = encode(queries, is_query=True)     # (200, 768)

    print(f"{'dim':>5} {'Recall@10':>10}")
    for d in TRUNC_DIMS:
        D, Q = truncate(D_full, d), truncate(Q_full, d)
        r = recall_at_k(D, Q, gold, TOP_K)
        print(f"{d:>5} {r:>10.3f}")

if __name__ == "__main__":
    main()
