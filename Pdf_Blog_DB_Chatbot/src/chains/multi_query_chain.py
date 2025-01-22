from retrievers.query_generator import generate_queries
from retrievers.rag_fusion import reciprocal_rank_fusion
from retrievers.retriever import batch_fetch_documents
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("""
Answer the following question based on this context:

{context}

Question: {question}
""")

llm = ChatOpenAI(temperature=0)

def multi_query_qa(input_query):
    """End-to-end multi-query QA with RAG-Fusion."""
    # Generate queries
    queries = generate_queries(input_query)
    # Fetch documents for each query
    results = batch_fetch_documents(queries)
    # Rerank using RAG-Fusion
    fused_results = reciprocal_rank_fusion(results)
    # Format context for final QA
    context = "\n\n".join([doc.page_content for doc in fused_results])
    # Generate final answer
    formatted_prompt = prompt.invoke({"context": context, "question": input_query})
    return llm.invoke(formatted_prompt)
