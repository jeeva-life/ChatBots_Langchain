def reciprocal_rank_fusion(results: list[list], k=60):
    """RRF combines multiple ranked lists into a unified ranking."""
    fused_scores = {}
    documents = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content  # Unique document identifier
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            fused_scores[doc_str] += 1 / (rank + k)  # RRF formula

    reranked_doc_strs = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)
    return [documents[doc_str] for doc_str in reranked_doc_strs]
