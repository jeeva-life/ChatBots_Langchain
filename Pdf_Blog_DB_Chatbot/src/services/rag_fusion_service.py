from chains.multi_query_chain import multi_query_qa

def get_answer_with_rag_fusion(question):
    """Handles the RAG-Fusion process and returns the final answer."""
    try:
        return multi_query_qa(question)
    except Exception as e:
        return {"error": str(e)}
