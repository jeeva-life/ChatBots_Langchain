def fetch_documents(query):
    """Retrieve documents based on a single query."""
    # Replace this with your actual retriever logic
    return retriever.retrieve(query)

def batch_fetch_documents(queries):
    """Fetch documents for all queries."""
    return [fetch_documents(query) for query in queries]
