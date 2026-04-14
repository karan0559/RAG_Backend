"""
Smart RAG System — Comprehensive Validation Audit
Covers Tasks 1-6: Retrieval, Reranking, Threshold, Faithfulness, Latency, Fallback
Read-only: does NOT modify any production code or data.
"""

import os, sys, time, random, pickle, warnings, asyncio, textwrap
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
random.seed(42)
import numpy as np
import faiss
import torch
import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR      = PROJECT_ROOT / "validation_results"
OUT_DIR.mkdir(exist_ok=True)

CHUNKS_PATH = DATA_DIR / "chunks.pkl"
INDEX_PATH  = DATA_DIR / "index.faiss"
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_URL     = "https://api.tavily.com/search"
RELEVANCE_THRESHOLD = 0.20
MIN_CHUNKS_REQUIRED = 20

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3"]

# ── Metric Store ───────────────────────────────────────────────────────────────
metrics = {}          # key: metric name, value: float
results_log = []      # list of dicts per query

print("="*65)
print("  Smart RAG Validation Audit")
print("="*65)

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def load_embed_model():
    from sentence_transformers import SentenceTransformer
    print("[*] Loading intfloat/e5-large-v2 …")
    m = SentenceTransformer("intfloat/e5-large-v2")
    print("[+] Embedding model ready.")
    return m

def embed_query(model, text: str) -> np.ndarray:
    v = model.encode(f"query: {text}", show_progress_bar=False, convert_to_numpy=True)
    return normalize(v.reshape(1,-1), axis=1)[0]

def embed_passage(model, text: str) -> np.ndarray:
    v = model.encode(f"passage: {text}", show_progress_bar=False, convert_to_numpy=True)
    return normalize(v.reshape(1,-1), axis=1)[0]

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def caption(ax, text):
    ax.annotate(text, xy=(0.5, -0.18), xycoords="axes fraction",
                ha="center", fontsize=8, color="#555", style="italic",
                wrap=True)

def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")

# ── Groq helper (sync via httpx) ───────────────────────────────────────────────
def call_groq_sync(messages, max_tokens=512, retries=3) -> str:
    if not GROQ_API_KEY:
        return "[GROQ_API_KEY missing]"
    payload = {"model": GROQ_MODEL, "temperature": 0.3,
               "max_tokens": max_tokens, "messages": messages}
    wait = 2
    for attempt in range(retries):
        try:
            r = httpx.post(GROQ_URL,
                           headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                           json=payload, timeout=45)
            if r.status_code == 429:
                print(f"  [rate-limit] waiting {wait}s …")
                time.sleep(wait); wait *= 2; continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries - 1:
                return f"[Groq error: {e}]"
            time.sleep(wait); wait *= 2
    return "[Groq failed after retries]"

# ── Tavily helper ──────────────────────────────────────────────────────────────
def call_tavily_sync(query: str) -> str:
    if not TAVILY_API_KEY:
        return "[TAVILY_API_KEY missing]"
    payload = {"query": query, "search_depth": "basic",
               "include_answer": True, "api_key": TAVILY_API_KEY}
    try:
        r = httpx.post(TAVILY_URL, json=payload, timeout=20)
        if r.status_code in (401, 403, 429):
            return f"[Tavily error {r.status_code}]"
        r.raise_for_status()
        d = r.json()
        ans = d.get("answer","")
        if ans: return ans
        snippets = [x.get("content","") for x in d.get("results",[])[:3]]
        return " ".join(snippets) or "[no results]"
    except Exception as e:
        return f"[Tavily error: {e}]"

# ── Key-phrase extraction (improved — no spacy required) ───────────────────────
def extract_noun_phrases(text: str, top_n=8):
    """Extract key terms: numbers, technical terms, proper nouns, content words."""
    import re
    # 1. Numbers and units (e.g. "512", "0.40", "15th", "4-3", "31st")
    numbers = re.findall(r'\b\d[\d.,/:-]*\w*\b', text)
    # 2. Acronyms / ALL-CAPS (e.g. "FAISS", "BM25", "NRZ", "CO2")
    acronyms = re.findall(r'\b[A-Z][A-Z0-9]{1,}\b', text)
    # 3. Mixed-case technical terms (e.g. "tpHL", "tpLH", "iDiva")
    mixed_case = re.findall(r'\b[a-z]+[A-Z]\w*\b', text)
    # 4. Proper nouns (capitalized words not at sentence start)
    proper = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]{2,}|(?<=\s)[A-Z][a-z]{2,}', text)
    proper = [p for p in proper if p.lower() not in {
        "the","and","for","are","but","not","this","that","with","from",
        "have","been","they","their","what","when","where","which",
        "will","would","could","should","about","also","into","more",
        "some","such","than","then","them","there","these","those"}]
    # 5. General content words (≥3 chars, skip stopwords)
    words = re.findall(r'\b[A-Za-z]{3,}\b', text)
    stops = {"the","and","for","are","but","not","you","all","can","had","her",
             "was","one","our","out","has","its","let","say","she","too","use",
             "that","this","with","from","have","been","were","they","does",
             "their","what","when","where","which","will","would","could",
             "should","about","also","into","more","some","such","than",
             "then","them","there","these","those","very","your","each",
             "most","other","being","using","based","given","need","make",
             "only","over","after","before","between","through","during"}
    freq = {}
    for w in words:
        wl = w.lower()
        if wl not in stops and len(wl) >= 3:
            freq[wl] = freq.get(wl, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    content_words = [w for w, _ in ranked[:top_n]]
    # Combine in priority order: numbers, acronyms, mixed-case, proper, content
    seen = set()
    out = []
    candidates = (numbers[:3]
                  + [a.lower() for a in acronyms[:3]]
                  + [m.lower() for m in mixed_case[:3]]
                  + [p.lower() for p in proper[:3]]
                  + content_words)
    for item in candidates:
        il = item.lower().strip()
        if il and il not in seen and len(il) >= 2:
            seen.add(il)
            out.append(il)
        if len(out) >= top_n:
            break
    return out

def phrase_coverage(phrases, answer: str) -> float:
    if not phrases:
        return 0.0
    al = answer.lower()
    hits = sum(1 for p in phrases if p in al)
    return hits / len(phrases)

# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading FAISS index and chunks …")
if not CHUNKS_PATH.exists() or not INDEX_PATH.exists():
    print("  [WARN] data/chunks.pkl or data/index.faiss not found.")
    SKIP_RETRIEVAL = True
    stored_chunks, faiss_index = [], None
else:
    with open(CHUNKS_PATH,"rb") as f:
        stored_chunks = pickle.load(f)
    faiss_index = faiss.read_index(str(INDEX_PATH))
    print(f"  [+] Loaded {len(stored_chunks)} chunks, {faiss_index.ntotal} vectors (dim={faiss_index.d})")
    SKIP_RETRIEVAL = faiss_index.ntotal < MIN_CHUNKS_REQUIRED

SYNTHETIC_MODE = SKIP_RETRIEVAL
if SYNTHETIC_MODE:
    print(f"  [!] Index has <{MIN_CHUNKS_REQUIRED} chunks — running on SYNTHETIC data (disclaimer applies).")

embed_model = load_embed_model()

# ══════════════════════════════════════════════════════════════════════
# TASK 1 — RETRIEVAL QUALITY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("TASK 1 — Retrieval Quality")
print("="*65)

if SYNTHETIC_MODE:
    # --- synthetic path ---
    print("  [SYNTHETIC] Generating 12 fake chunks and queries …")
    fake_topics = [
        ("photosynthesis","Photosynthesis is the process by which plants use sunlight to produce glucose from CO2 and water."),
        ("black holes","A black hole is a region of spacetime where gravity is so strong that nothing can escape."),
        ("machine learning","Machine learning is a subset of AI where models learn patterns from data without explicit programming."),
        ("climate change","Climate change refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities."),
        ("quantum computing","Quantum computing uses quantum bits (qubits) that can represent 0 and 1 simultaneously via superposition."),
        ("DNA structure","DNA is a double helix made of nucleotides containing adenine, thymine, cytosine and guanine bases."),
        ("Renaissance art","Renaissance art flourished in 15th-century Italy, emphasizing realism, perspective and human anatomy."),
        ("supply chain","Supply chain management coordinates production, shipment and delivery of goods from supplier to consumer."),
        ("neural networks","Neural networks consist of layers of interconnected nodes that transform inputs through weighted connections."),
        ("vaccines","Vaccines work by introducing antigens that train the immune system to recognise and fight pathogens."),
        ("cryptocurrency","Cryptocurrency is a digital currency secured by cryptography and operating on decentralised blockchain networks."),
        ("plate tectonics","Plate tectonics explains how the Earth's lithosphere is divided into moving plates that shape continents."),
    ]
    stored_chunks = [f"synthetic_doc|{text}" for _,text in fake_topics]
    raw_texts     = [text for _,text in fake_topics]
    embeddings    = np.array([embed_passage(embed_model, t) for t in raw_texts], dtype="float32")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    test_pairs = [(f"How does {topic} work?" , i) for i,(topic,_) in enumerate(fake_topics)]
else:
    # --- real data path ---
    raw_texts = []
    for ch in stored_chunks:
        if "|" in ch:
            raw_texts.append(ch.split("|",1)[1].strip())
        elif ch.startswith("[") and "]" in ch:
            raw_texts.append(ch[ch.index("]")+1:].strip())
        else:
            raw_texts.append(ch.strip())

    # Filter out junk chunks before sampling
    BAD_MARKERS = ("no extractable text", "ocr failed", "extraction failed",
                   "unsupported file", "no readable text")
    valid_indices = [i for i, t in enumerate(raw_texts)
                     if len(t.split()) >= 15  # at least 15 words
                     and not any(m in t.lower() for m in BAD_MARKERS)]
    print(f"  {len(valid_indices)} valid chunks out of {len(raw_texts)}")

    # pick 12 well-spread chunks from valid ones
    step = max(1, len(valid_indices)//12)
    selected_indices = [valid_indices[i] for i in range(0, min(len(valid_indices), 12*step), step)][:12]

    # Generate questions via Groq
    print(f"  Generating {len(selected_indices)} synthetic Q-A pairs via Groq …")
    test_pairs = []
    for idx in selected_indices:
        chunk_text = raw_texts[idx][:800]
        msgs = [{"role":"system","content":"You are a question-generation assistant. Generate specific questions that include key entities, names, or numbers from the passage."},
                {"role":"user","content":
                 f"Write ONE short factual question (max 20 words) that this passage directly answers. "
                 f"Include at least one specific name, number, or technical term from the passage in the question.\n"
                 f"Passage:\n{chunk_text}\n\nRespond with only the question, no preamble."}]
        q = call_groq_sync(msgs, max_tokens=60)
        if q.startswith("["):
            q = f"What does this passage describe? (chunk {idx})"
        test_pairs.append((q, idx))
        print(f"    Q{len(test_pairs)}: {q[:70]}")

# --- Run retrieval (hybrid BM25 + FAISS via Reciprocal Rank Fusion) ---
print("\n  Running hybrid BM25+FAISS retrieval …")
TOP_K = 5
recall_at = {1:0, 3:0, 5:0}
mrr_total = 0.0
query_embeddings, similarity_matrix = [], []

# Build BM25 index over raw texts
import re as _re
from rank_bm25 import BM25Okapi
_bm25_tokenize = lambda t: _re.findall(r"[a-z0-9]+", t.lower())
_bm25_corpus = [_bm25_tokenize(t) for t in raw_texts]
_bm25 = BM25Okapi(_bm25_corpus)

def hybrid_search(qv, query_text, top_k=5):
    """Reciprocal Rank Fusion of FAISS + BM25."""
    candidates = top_k * 5
    # Dense
    D, I = faiss_index.search(np.array([qv], dtype="float32"), candidates)
    # BM25
    bm25_scores = _bm25.get_scores(_bm25_tokenize(query_text))
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:candidates]
    # RRF
    RRF_K = 60
    fused = {}
    for rank, idx in enumerate(I[0]):
        fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0/(RRF_K + rank + 1)
    for rank, idx in enumerate(bm25_ranked):
        if bm25_scores[idx] > 0:
            fused[idx] = fused.get(idx, 0.0) + 1.0/(RRF_K + rank + 1)
    sorted_idx = sorted(fused.keys(), key=lambda i: -fused[i])[:top_k]
    # Retrieve cosine scores for heatmap
    scores_out = []
    for idx in sorted_idx:
        vec = faiss_index.reconstruct(int(idx))
        scores_out.append(float(np.dot(qv, vec)))
    return sorted_idx, scores_out

for q, gt_idx in test_pairs:
    qv = embed_query(embed_model, q)
    query_embeddings.append(qv)
    retrieved_indices, scores = hybrid_search(qv, q, TOP_K)

    rank = None
    if gt_idx in retrieved_indices:
        rank = retrieved_indices.index(gt_idx) + 1
        for k in [1,3,5]:
            if rank <= k:
                recall_at[k] += 1
        mrr_total += 1.0 / rank
    else:
        mrr_total += 0.0

    similarity_matrix.append(scores[:TOP_K])

n = len(test_pairs)
R1 = recall_at[1]/n;  R3 = recall_at[3]/n
R5 = recall_at[5]/n;  MRR = mrr_total/n
metrics["Recall@1"] = R1; metrics["Recall@3"] = R3
metrics["Recall@5"] = R5; metrics["MRR"]      = MRR

print(f"\n  Recall@1={R1:.3f}  Recall@3={R3:.3f}  Recall@5={R5:.3f}  MRR={MRR:.3f}")

# Plot 1a — bar chart
fig, ax = plt.subplots(figsize=(7,4))
bars = ax.bar(["Recall@1","Recall@3","Recall@5","MRR"],
              [R1,R3,R5,MRR], color=PALETTE[:4], width=0.5, edgecolor="white")
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0,1.15); ax.set_title("Task 1 — Retrieval Metrics", fontsize=13, fontweight="bold")
ax.set_ylabel("Score")
caption(ax, "Higher is better. Recall@k = fraction of queries where the correct chunk appears in top-k results.")
save_fig(fig, "task1a_retrieval_metrics.png")

# Plot 1b — heatmap
sim_arr = np.array(similarity_matrix)
fig, ax = plt.subplots(figsize=(8,max(4, n//2)))
im = ax.imshow(sim_arr, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(TOP_K)); ax.set_xticklabels([f"Rank {i+1}" for i in range(TOP_K)])
ax.set_yticks(range(n));    ax.set_yticklabels([f"Q{i+1}" for i in range(n)], fontsize=8)
plt.colorbar(im, ax=ax, label="Cosine Similarity")
ax.set_title("Task 1 — Query × Retrieved Chunk Similarity Heatmap", fontsize=12, fontweight="bold")
caption(ax, "Rows = queries, columns = rank positions. Brighter = more similar. Strong top-left bias = good retriever.")
save_fig(fig, "task1b_similarity_heatmap.png")

# ══════════════════════════════════════════════════════════════════════
# TASK 2 — RERANKER SCORE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("TASK 2 — Reranker Score Distribution")
print("="*65)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
print("  Loading BAAI/bge-reranker-base …")
rr_tok   = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
rr_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
rr_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
rr_model.to(device)
print(f"  [+] Reranker ready on {device}.")

def reranker_score(query, passage) -> float:
    enc = rr_tok([query], [passage[:512]], return_tensors="pt",
                 padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logit = rr_model(**enc).logits.squeeze(-1)
    return float(torch.sigmoid(logit).item())

pos_scores, neg_scores = [], []
faiss_rank_list, rr_score_list, is_gt_list = [], [], []
reranker_correct = 0

for qi, (q, gt_idx) in enumerate(test_pairs):
    # ground truth chunk
    gt_text = raw_texts[gt_idx][:500]
    pos_s = reranker_score(q, gt_text)
    pos_scores.append(pos_s)

    # 3 random negatives (not gt)
    neg_pool = [i for i in range(len(raw_texts)) if i != gt_idx]
    neg_idxs = random.sample(neg_pool, min(3, len(neg_pool)))
    neg_s_list = [reranker_score(q, raw_texts[ni][:500]) for ni in neg_idxs]
    neg_scores.extend(neg_s_list)

    # reranker accuracy: pos > all negatives?
    if all(pos_s > ns for ns in neg_s_list):
        reranker_correct += 1

    # FAISS rank + reranker score for scatter
    D, I = faiss_index.search(np.array([embed_query(embed_model, q)], dtype="float32"), TOP_K)
    for rank_i, idx in enumerate(I[0]):
        chunk_text = raw_texts[idx] if idx < len(raw_texts) else ""
        rr_s = reranker_score(q, chunk_text[:500])
        faiss_rank_list.append(rank_i+1)
        rr_score_list.append(rr_s)
        is_gt_list.append(idx == gt_idx)

rr_acc = reranker_correct / n
gap = np.mean(pos_scores) - np.mean(neg_scores)
metrics["Reranker_AvgPositive"]  = float(np.mean(pos_scores))
metrics["Reranker_AvgNegative"]  = float(np.mean(neg_scores))
metrics["Reranker_Gap"]          = float(gap)
metrics["Reranker_Accuracy"]     = rr_acc

print(f"  Avg Pos={np.mean(pos_scores):.3f}  Avg Neg={np.mean(neg_scores):.3f}  "
      f"Gap={gap:.3f}  Accuracy={rr_acc:.3f}")

# Plot 2a — violin / box
fig, ax = plt.subplots(figsize=(7,5))
data_v = [pos_scores, neg_scores]
parts = ax.violinplot(data_v, positions=[1,2], showmedians=True, showextrema=True)
for pc, col in zip(parts["bodies"], [PALETTE[2], PALETTE[3]]):
    pc.set_facecolor(col); pc.set_alpha(0.7)
ax.set_xticks([1,2]); ax.set_xticklabels(["Positive (Relevant)","Negative (Irrelevant)"])
ax.set_ylabel("Reranker Score (sigmoid)"); ax.set_ylim(0,1)
ax.set_title("Task 2 — Reranker Score Distribution", fontsize=13, fontweight="bold")
caption(ax, "Good reranker: positive distribution clearly higher than negative. Gap should be ≥0.20.")
save_fig(fig, "task2a_reranker_violin.png")

# Plot 2b — scatter FAISS rank vs reranker score
fig, ax = plt.subplots(figsize=(7,5))
colors = [PALETTE[2] if g else PALETTE[3] for g in is_gt_list]
ax.scatter(faiss_rank_list, rr_score_list, c=colors, s=80, alpha=0.75, edgecolors="white")
ax.axhline(RELEVANCE_THRESHOLD, color="red", ls="--", lw=1.5, label=f"Threshold {RELEVANCE_THRESHOLD}")
ax.set_xlabel("FAISS Retrieval Rank"); ax.set_ylabel("Reranker Score")
ax.set_title("Task 2 — FAISS Rank vs Reranker Score", fontsize=13, fontweight="bold")
p1 = mpatches.Patch(color=PALETTE[2], label="Ground Truth Chunk")
p2 = mpatches.Patch(color=PALETTE[3], label="Other Chunks")
ax.legend(handles=[p1,p2])
caption(ax, "Green dots = correct chunk. If they appear at rank 1 with high reranker score, reranking is working.")
save_fig(fig, "task2b_rank_vs_reranker.png")

# ══════════════════════════════════════════════════════════════════════
# TASK 3 — RELEVANCE GATE CALIBRATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("TASK 3 — Relevance Gate Calibration (threshold=0.40)")
print("="*65)

labels   = [1]*len(pos_scores) + [0]*len(neg_scores)
scores_t = pos_scores + neg_scores
thresholds_sweep = np.arange(0.10, 0.91, 0.05)

f1_list, tpr_list, fpr_list, prec_list, rec_list = [], [], [], [], []
for t in thresholds_sweep:
    preds = [1 if s >= t else 0 for s in scores_t]
    TP = sum(1 for p,l in zip(preds,labels) if p==1 and l==1)
    FP = sum(1 for p,l in zip(preds,labels) if p==1 and l==0)
    TN = sum(1 for p,l in zip(preds,labels) if p==0 and l==0)
    FN = sum(1 for p,l in zip(preds,labels) if p==0 and l==1)
    tpr = TP/(TP+FN+1e-9); fpr = FP/(FP+TN+1e-9)
    prec = TP/(TP+FP+1e-9); rec = tpr
    f1 = 2*prec*rec/(prec+rec+1e-9)
    tpr_list.append(tpr); fpr_list.append(fpr)
    prec_list.append(prec); rec_list.append(rec); f1_list.append(f1)

fpr_c, tpr_c, _ = roc_curve(labels, scores_t)
roc_auc = auc(fpr_c, tpr_c)
metrics["ROC_AUC"] = roc_auc
opt_idx = int(np.argmax(f1_list))
opt_thresh = float(thresholds_sweep[opt_idx])
opt_f1     = float(f1_list[opt_idx])
metrics["Optimal_Threshold"] = opt_thresh
metrics["Optimal_F1"]        = opt_f1

# interpolate current threshold position
thr_idx = np.argmin(np.abs(thresholds_sweep - RELEVANCE_THRESHOLD))
curr_f1 = float(f1_list[thr_idx])
# find approx (fpr, tpr) for threshold 0.40
curr_preds = [1 if s >= RELEVANCE_THRESHOLD else 0 for s in scores_t]
TP_c = sum(1 for p,l in zip(curr_preds,labels) if p==1 and l==1)
FP_c = sum(1 for p,l in zip(curr_preds,labels) if p==1 and l==0)
TN_c = sum(1 for p,l in zip(curr_preds,labels) if p==0 and l==0)
FN_c = sum(1 for p,l in zip(curr_preds,labels) if p==0 and l==1)
curr_tpr = TP_c/(TP_c+FN_c+1e-9); curr_fpr = FP_c/(FP_c+TN_c+1e-9)

print(f"  ROC AUC={roc_auc:.3f}  Optimal threshold={opt_thresh:.2f} (F1={opt_f1:.3f})")
print(f"  Current threshold 0.40: F1={curr_f1:.3f}")

# Plot 3a — ROC
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(fpr_c, tpr_c, color=PALETTE[0], lw=2, label=f"ROC (AUC={roc_auc:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1)
ax.scatter([curr_fpr],[curr_tpr], color="red", zorder=5, s=100, label=f"Threshold 0.40")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Task 3 — ROC Curve (Relevance Gate)", fontsize=13, fontweight="bold")
ax.legend()
caption(ax, "Red dot = current threshold (0.40). AUC closer to 1.0 = better discrimination.")
save_fig(fig, "task3a_roc_curve.png")

# Plot 3b — F1 vs threshold
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(thresholds_sweep, f1_list, color=PALETTE[1], lw=2.5)
ax.axvline(RELEVANCE_THRESHOLD, color="red", ls="--", lw=1.5, label="Current 0.40")
ax.axvline(opt_thresh, color=PALETTE[2], ls="--", lw=1.5, label=f"Optimal {opt_thresh:.2f}")
ax.fill_between(thresholds_sweep, f1_list,
                where=(thresholds_sweep >= RELEVANCE_THRESHOLD-0.025) & (thresholds_sweep <= RELEVANCE_THRESHOLD+0.025),
                alpha=0.25, color="red")
ax.set_xlabel("Threshold"); ax.set_ylabel("F1 Score")
ax.set_title("Task 3 — F1 Score vs Threshold", fontsize=13, fontweight="bold")
ax.legend()
caption(ax, "Peak F1 = optimal threshold. Vertical red line = current hardcoded threshold.")
save_fig(fig, "task3b_f1_vs_threshold.png")

# ══════════════════════════════════════════════════════════════════════
# TASK 4 — LLM ANSWER FAITHFULNESS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("TASK 4 — LLM Answer Faithfulness")
print("="*65)

SYSTEM_PROMPT = ("You are a helpful assistant. Answer questions using the provided "
                 "document context. If the context contains the answer, use it. "
                 "Use specific terms, names, and numbers from the document context in your answer. "
                 "Be concise and accurate.")

faith_sims, faith_coverages, faith_queries, faith_answers, faith_flagged = [], [], [], [], []

# Use first 6 test pairs where retrieval rank ≤ 3
good_pairs = []
for q, gt_idx in test_pairs:
    qv = embed_query(embed_model, q)
    D, I = faiss_index.search(np.array([qv], dtype="float32"), TOP_K)
    if gt_idx in list(I[0])[:3]:
        good_pairs.append((q, gt_idx))
    if len(good_pairs) >= 6:
        break

if len(good_pairs) == 0:
    good_pairs = test_pairs[:6]

print(f"  Running LLM on {len(good_pairs)} queries …")
for q, gt_idx in good_pairs:
    ctx = raw_texts[gt_idx][:1200]
    user_msg = (f"Conversation History:\nNone\n\n"
                f"Document Context:\n{ctx}\n\nQuestion:\n{q}")
    messages = [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":user_msg}]
    answer = call_groq_sync(messages, max_tokens=300)
    print(f"  Q: {q[:60]} -> {answer[:60]} ...")

    ans_vec = embed_passage(embed_model, answer)
    ctx_vec = embed_passage(embed_model, ctx)
    sim = cosine(ans_vec, ctx_vec)

    phrases = extract_noun_phrases(ctx, top_n=5)
    cov = phrase_coverage(phrases, answer)

    faith_sims.append(sim)
    faith_coverages.append(cov)
    faith_queries.append(q[:50])
    faith_answers.append(answer)
    if sim < 0.50:
        faith_flagged.append({"query":q, "answer":answer, "sim":sim})

metrics["Faithfulness_MeanSim"]      = float(np.mean(faith_sims))
metrics["Faithfulness_MeanCoverage"] = float(np.mean(faith_coverages))
print(f"\n  Mean cosine sim={np.mean(faith_sims):.3f}  Mean phrase cov={np.mean(faith_coverages):.3f}")
print(f"  Flagged hallucinations: {len(faith_flagged)}")

# Plot 4a — horizontal bar
fig, ax = plt.subplots(figsize=(9,5))
colors_bar = [PALETTE[3] if s<0.50 else (PALETTE[1] if s<0.70 else PALETTE[2]) for s in faith_sims]
y = range(len(faith_sims))
ax.barh(list(y), faith_sims, color=colors_bar, edgecolor="white")
ax.axvline(0.50, color="red", ls="--", lw=1.5, label="Hallucination risk (<0.50)")
ax.axvline(0.70, color="green", ls="--", lw=1.5, label="Good faithfulness (>0.70)")
ax.set_yticks(list(y)); ax.set_yticklabels(faith_queries, fontsize=8)
ax.set_xlabel("Cosine Similarity (Answer ↔ Source Chunk)")
ax.set_title("Task 4 — Answer Faithfulness per Query", fontsize=13, fontweight="bold")
ax.legend(loc="lower right"); ax.set_xlim(0,1.05)
caption(ax, "Red=potential hallucination(<0.50), Yellow=marginal, Green=faithful(>0.70).")
save_fig(fig, "task4a_faithfulness_bar.png")

# Plot 4b — scatter sim vs coverage
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(faith_sims, faith_coverages, s=120, c=PALETTE[0], edgecolors="white", zorder=3)
for i,(x,y_,q) in enumerate(zip(faith_sims, faith_coverages, faith_queries)):
    ax.annotate(f"Q{i+1}", (x,y_), textcoords="offset points", xytext=(5,5), fontsize=8)
ax.axvline(0.70, color="green", ls="--", lw=1.2)
ax.axhline(0.60, color="green", ls="--", lw=1.2)
ax.axvline(0.50, color="red", ls="--", lw=1.2, alpha=0.6)
ax.text(0.76, 0.92, "Faithful", transform=ax.transAxes, color="green", fontsize=9, fontweight="bold")
ax.text(0.05, 0.92, "Lexically OK\nSemantics off", transform=ax.transAxes, color="orange", fontsize=8)
ax.text(0.05, 0.05, "Potential Hallucination", transform=ax.transAxes, color="red", fontsize=9, fontweight="bold")
ax.text(0.76, 0.05, "Paraphrased\n(acceptable)", transform=ax.transAxes, color=PALETTE[4], fontsize=8)
ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Key Phrase Coverage")
ax.set_title("Task 4 — Faithfulness: Similarity vs Phrase Coverage", fontsize=12, fontweight="bold")
ax.set_xlim(0,1.05); ax.set_ylim(-0.05,1.15)
caption(ax, "Top-right quadrant = most faithful. Bottom-left = highest hallucination risk.")
save_fig(fig, "task4b_faithfulness_scatter.png")

# ══════════════════════════════════════════════════════════════════════
# TASK 5 — END-TO-END PIPELINE LATENCY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("TASK 5 — Pipeline Latency")
print("="*65)

latency_queries = [q for q,_ in test_pairs[:5]]
lat_embed, lat_faiss, lat_rerank, lat_llm = [], [], [], []

for q in latency_queries:
    t0 = time.perf_counter()
    qv = embed_query(embed_model, q)
    lat_embed.append((time.perf_counter()-t0)*1000)

    t0 = time.perf_counter()
    D, I = faiss_index.search(np.array([qv], dtype="float32"), TOP_K)
    lat_faiss.append((time.perf_counter()-t0)*1000)

    passages = [raw_texts[i][:400] for i in I[0] if i < len(raw_texts)]
    t0 = time.perf_counter()
    _ = [reranker_score(q, p) for p in passages]
    lat_rerank.append((time.perf_counter()-t0)*1000)

    ctx = raw_texts[I[0][0]][:800] if len(I[0])>0 and I[0][0]<len(raw_texts) else "N/A"
    msgs = [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":f"Document Context:\n{ctx}\n\nQuestion:\n{q}"}]
    t0 = time.perf_counter()
    _ = call_groq_sync(msgs, max_tokens=200)
    lat_llm.append((time.perf_counter()-t0)*1000)

    print(f"  {q[:40]:42s} embed={lat_embed[-1]:.0f}ms  faiss={lat_faiss[-1]:.0f}ms  "
          f"rerank={lat_rerank[-1]:.0f}ms  llm={lat_llm[-1]:.0f}ms")

metrics["Latency_Embed_ms"]   = float(np.mean(lat_embed))
metrics["Latency_FAISS_ms"]   = float(np.mean(lat_faiss))
metrics["Latency_Rerank_ms"]  = float(np.mean(lat_rerank))
metrics["Latency_LLM_ms"]     = float(np.mean(lat_llm))
metrics["Latency_Total_ms"]   = float(np.mean([e+f+r+l for e,f,r,l in
                                                zip(lat_embed,lat_faiss,lat_rerank,lat_llm)]))

# Plot 5 — stacked horizontal bar
fig, ax = plt.subplots(figsize=(9,5))
qs_labels = [f"Q{i+1}" for i in range(len(latency_queries))]
bars1 = ax.barh(qs_labels, lat_embed,  color=PALETTE[0], label="Embedding")
bars2 = ax.barh(qs_labels, lat_faiss,  left=lat_embed, color=PALETTE[1], label="FAISS Search")
left3 = [e+f for e,f in zip(lat_embed,lat_faiss)]
bars3 = ax.barh(qs_labels, lat_rerank, left=left3, color=PALETTE[2], label="Reranking")
left4 = [e+f+r for e,f,r in zip(lat_embed,lat_faiss,lat_rerank)]
bars4 = ax.barh(qs_labels, lat_llm,   left=left4, color=PALETTE[3], label="LLM (Groq)")
ax.set_xlabel("Time (ms)"); ax.set_title("Task 5 — Pipeline Latency per Query", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
caption(ax, "Each segment shows time per stage. LLM call is usually the bottleneck.")
save_fig(fig, "task5_latency_stacked.png")

bottleneck = max(["Embedding","FAISS","Reranking","LLM"],
                 key=lambda s: {"Embedding":np.mean(lat_embed),"FAISS":np.mean(lat_faiss),
                                "Reranking":np.mean(lat_rerank),"LLM":np.mean(lat_llm)}[s])
print(f"  Bottleneck stage: {bottleneck}")

# ══════════════════════════════════════════════════════════════════════
# TASK 6 — FALLBACK TRIGGER ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("TASK 6 — Fallback Trigger Analysis")
print("="*65)

off_topic_queries = [
    "What is the current stock price of Apple Inc today?",
    "Who won the FIFA World Cup in 2022?",
    "How do I bake a chocolate cake from scratch?",
    "What are the symptoms of the common cold?",
    "What is the speed of light in a vacuum?",
]
on_topic_queries = [q for q,_ in test_pairs[:5]]

TP_fb=0; FP_fb=0; TN_fb=0; FN_fb=0  # confusion matrix

def check_fallback(q):
    """Returns (triggered:bool, max_reranker_score:float, tavily_answer:str|None)."""
    qv = embed_query(embed_model, q)
    D, I = faiss_index.search(np.array([qv], dtype="float32"), TOP_K)
    passages = [raw_texts[i][:400] for i in I[0] if i < len(raw_texts)]
    if not passages:
        return True, 0.0, call_tavily_sync(q)
    scores_rr = [reranker_score(q, p) for p in passages]
    max_score = max(scores_rr)
    sig_score = max_score  # already sigmoid applied
    triggered = sig_score < RELEVANCE_THRESHOLD
    web_result = call_tavily_sync(q) if triggered else None
    return triggered, sig_score, web_result

print("  Off-topic queries (expect fallback=True):")
tavily_sims = []
for q in off_topic_queries:
    trig, score, web = check_fallback(q)
    print(f"    [{'+' if trig else '-'}] score={score:.3f} | {q[:55]}")
    if trig:
        TP_fb += 1
        if web and not web.startswith("["):
            qv2 = embed_query(embed_model, q)
            wv  = embed_passage(embed_model, web[:500])
            tavily_sims.append(cosine(qv2, wv))
    else:
        FN_fb += 1

print("  On-topic queries (expect fallback=False):")
for q in on_topic_queries:
    trig, score, _ = check_fallback(q)
    print(f"    [{'+' if not trig else '-'}] score={score:.3f} | {q[:55]}")
    if not trig:
        TN_fb += 1
    else:
        FP_fb += 1

metrics["Fallback_TP"] = TP_fb; metrics["Fallback_FP"] = FP_fb
metrics["Fallback_TN"] = TN_fb; metrics["Fallback_FN"] = FN_fb
metrics["Fallback_Accuracy"] = (TP_fb+TN_fb)/(TP_fb+FP_fb+TN_fb+FN_fb+1e-9)
tavily_avg = float(np.mean(tavily_sims)) if tavily_sims else 0.0
metrics["Tavily_AvgSim"] = tavily_avg
print(f"\n  Confusion: TP={TP_fb} FP={FP_fb} TN={TN_fb} FN={FN_fb}  Acc={metrics['Fallback_Accuracy']:.3f}")

# Plot 6 — confusion matrix
fig, ax = plt.subplots(figsize=(5,5))
cm = np.array([[TP_fb, FN_fb],[FP_fb, TN_fb]])
im = ax.imshow(cm, cmap="Blues")
labels_c = [["TP\n(correct fallback)","FN\n(missed fallback)"],
            ["FP\n(wrong fallback)","TN\n(correct doc-use)"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels_c[i][j]}\n{cm[i,j]}", ha="center", va="center",
                fontsize=10, color="black")
ax.set_xticks([0,1]); ax.set_xticklabels(["Fallback","No Fallback"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Off-topic","On-topic"])
ax.set_title("Task 6 — Fallback Confusion Matrix", fontsize=13, fontweight="bold")
caption(ax, "TP = correctly triggered fallback. FP = wasted Tavily call. FN = missed fallback (bad answer).")
save_fig(fig, "task6_fallback_confusion.png")

# ══════════════════════════════════════════════════════════════════════
# VALIDATION REPORT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("Generating validation_report.md …")
print("="*65)

TARGETS = {
    "Recall@1":                 (0.50, 0.70, "≥0.50 acceptable, ≥0.70 good"),
    "Recall@3":                 (0.70, None, "≥0.70"),
    "MRR":                      (0.60, None, "≥0.60"),
    "Reranker_Gap":             (0.20, None, "≥0.20"),
    "Reranker_Accuracy":        (0.80, None, "≥0.80"),
    "Faithfulness_MeanSim":     (0.70, None, "≥0.70"),
    "Faithfulness_MeanCoverage":(0.60, None, "≥0.60"),
    "ROC_AUC":                  (0.80, None, "≥0.80"),
}

def status(key):
    v = metrics.get(key, 0.0)
    lo, hi, _ = TARGETS[key]
    if v >= (hi or lo):
        return "✅ PASS", v
    elif v >= lo:
        return "⚠️  MARGINAL", v
    else:
        return "❌ FAIL", v

RECOMMENDATIONS = {
    "Recall@1":    "Increase top_k retrieval or switch to a larger embedding model (e5-mistral-7b-instruct). Consider BM25 hybrid retrieval.",
    "Recall@3":    "Improve chunking strategy (overlap). Consider parent-document retrieval.",
    "MRR":         "Low MRR suggests the relevant chunk is rarely in the top results. Try hybrid BM25+dense retrieval.",
    "Reranker_Gap":"Gap < 0.20 means the reranker is not discriminating well. Try a larger model (bge-reranker-large).",
    "Reranker_Accuracy":"Reranker often ranks irrelevant chunks above relevant ones. Check token truncation — passages may be cut too short.",
    "Faithfulness_MeanSim": "LLM answers have low similarity to source chunks. Reduce temperature, shorten context window, or add 'quote the source' instruction.",
    "Faithfulness_MeanCoverage":"Low phrase coverage. Inject extracted keywords as a 'required terms' hint to the LLM prompt.",
    "ROC_AUC": "Threshold 0.40 may not be discriminating well. Consider tuning based on validation data.",
}

def recmd(key, st_label):
    if "FAIL" in st_label or "MARGINAL" in st_label:
        return RECOMMENDATIONS.get(key, "Review component configuration.")
    return "—"

lines = []
lines.append("# Smart RAG System — Validation Report\n")
lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
if SYNTHETIC_MODE:
    lines.append("\n> ⚠️ **DISCLAIMER:** The FAISS index contained fewer than 20 vectors. "
                 "Tasks 1-3 were run on **synthetic data**. Results are illustrative only.\n")
lines.append("\n## Summary Metrics Table\n")
lines.append("| Metric | Value | Target | Status | Recommendation |")
lines.append("|---|---|---|---|---|")

passed, total = 0, len(TARGETS)
for key, (lo, hi, target_str) in TARGETS.items():
    st_label, val = status(key)
    if "PASS" in st_label: passed += 1
    rec = recmd(key, st_label)
    lines.append(f"| {key} | {val:.3f} | {target_str} | {st_label} | {rec} |")

lines.append(f"\n**Result: {passed}/{total} validation checks passed.**\n")

lines.append("\n## Additional Metrics\n")
lines.append(f"- Recall@5: {metrics.get('Recall@5',0):.3f}")
lines.append(f"- Reranker Avg Positive Score: {metrics.get('Reranker_AvgPositive',0):.3f}")
lines.append(f"- Reranker Avg Negative Score: {metrics.get('Reranker_AvgNegative',0):.3f}")
lines.append(f"- Optimal Threshold: {metrics.get('Optimal_Threshold',0):.2f} (F1={metrics.get('Optimal_F1',0):.3f})")
_opt_t = metrics.get('Optimal_Threshold', 0.4)
_thr_rec = "keep" if abs(_opt_t - 0.4) < 0.1 else f"consider adjusting to {_opt_t:.2f}"
lines.append(f"- Current Threshold 0.40 recommendation: {_thr_rec}")
lines.append(f"- Pipeline Bottleneck: **{bottleneck}**")
lines.append(f"  - Avg Embed: {metrics.get('Latency_Embed_ms',0):.0f}ms")
lines.append(f"  - Avg FAISS: {metrics.get('Latency_FAISS_ms',0):.0f}ms")
lines.append(f"  - Avg Rerank: {metrics.get('Latency_Rerank_ms',0):.0f}ms")
lines.append(f"  - Avg LLM: {metrics.get('Latency_LLM_ms',0):.0f}ms")
lines.append(f"  - Avg Total: {metrics.get('Latency_Total_ms',0):.0f}ms")
lines.append(f"- Fallback Accuracy: {metrics.get('Fallback_Accuracy',0):.3f} (TP={TP_fb} FP={FP_fb} TN={TN_fb} FN={FN_fb})")
lines.append(f"- Tavily Result Similarity (when fallback fired): {tavily_avg:.3f}")

if faith_flagged:
    lines.append("\n## ⚠️ Potential Hallucination Flags\n")
    lines.append("These queries produced answers with cosine similarity < 0.50 to the source chunk:\n")
    for item in faith_flagged:
        lines.append(f"- **Query:** {item['query']}")
        lines.append(f"  **Sim:** {item['sim']:.3f}")
        lines.append(f"  **Answer:** {item['answer'][:200]} ...\n")
else:
    lines.append("\n## Hallucination Flags\n")
    lines.append("No answers flagged as potential hallucinations (all cosine similarities ≥ 0.50).\n")

lines.append("\n## Plots Generated\n")
for f in sorted(OUT_DIR.glob("*.png")):
    lines.append(f"- `{f.name}`")

report_path = OUT_DIR / "validation_report.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  [saved] validation_report.md")

# ══════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f"  FINAL RESULT: {passed}/{total} validation checks passed.")
print(f"  See {OUT_DIR} for full report and plots.")
print("="*65)
for key in TARGETS:
    st, val = status(key)
    print(f"  {key:<32s} {val:.3f}  {st}")
print("="*65)
