# Smart RAG System — Validation Report

**Generated:** 2026-04-14 12:03:14  

## Summary Metrics Table

| Metric | Value | Target | Status | Recommendation |
|---|---|---|---|---|
| Recall@1 | 0.750 | ≥0.50 acceptable, ≥0.70 good | ✅ PASS | — |
| Recall@3 | 0.917 | ≥0.70 | ✅ PASS | — |
| MRR | 0.854 | ≥0.60 | ✅ PASS | — |
| Reranker_Gap | 0.890 | ≥0.20 | ✅ PASS | — |
| Reranker_Accuracy | 0.917 | ≥0.80 | ✅ PASS | — |
| Faithfulness_MeanSim | 0.895 | ≥0.70 | ✅ PASS | — |
| Faithfulness_MeanCoverage | 0.200 | ≥0.60 | ❌ FAIL | Low phrase coverage. Inject extracted keywords as a 'required terms' hint to the LLM prompt. |
| ROC_AUC | 0.986 | ≥0.80 | ✅ PASS | — |

**Result: 7/8 validation checks passed.**


## Additional Metrics

- Recall@5: 1.000
- Reranker Avg Positive Score: 0.911
- Reranker Avg Negative Score: 0.021
- Optimal Threshold: 0.50 (F1=0.957)
- Current Threshold 0.40 recommendation: consider adjusting to 0.50
- Pipeline Bottleneck: **Reranking**
  - Avg Embed: 212ms
  - Avg FAISS: 1ms
  - Avg Rerank: 656ms
  - Avg LLM: 583ms
  - Avg Total: 1452ms
- Fallback Accuracy: 0.700 (TP=2 FP=0 TN=5 FN=3)
- Tavily Result Similarity (when fallback fired): 0.897

## Hallucination Flags

No answers flagged as potential hallucinations (all cosine similarities ≥ 0.50).


## Plots Generated

- `task1a_retrieval_metrics.png`
- `task1b_similarity_heatmap.png`
- `task2a_reranker_violin.png`
- `task2b_rank_vs_reranker.png`
- `task3a_roc_curve.png`
- `task3b_f1_vs_threshold.png`
- `task4a_faithfulness_bar.png`
- `task4b_faithfulness_scatter.png`
- `task5_latency_stacked.png`
- `task6_fallback_confusion.png`