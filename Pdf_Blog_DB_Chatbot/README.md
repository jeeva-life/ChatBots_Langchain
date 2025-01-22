# RAG Fusion-Based Multi-Query QA System

This project implements a robust Multi-Query Question Answering (QA) system using Retrieval-Augmented Generation (RAG) Fusion. The application is designed to answer user queries accurately by leveraging multiple retrievers and employing preventive measures against prompt injection attacks.

---

## Project Overview

The application performs the following key functions:
1. **Multi-Query Generation:** Generates multiple sub-queries from the input query to maximize context retrieval.
2. **RAG Fusion:** Combines the results from multiple retrievers using Reciprocal Rank Fusion for better accuracy.
3. **Prompt Injection Security:** Implements security measures to mitigate vulnerabilities related to generative AI prompts.
4. **Modular Design:** Ensures maintainability and scalability with a clean project structure.

---

## Project Structure

```plaintext
project-root/
│
├── chains/
│   └── multi_query_chain.py      # Multi-query RAG Fusion QA logic
│
├── retrievers/
│   ├── query_generator.py        # Generates multiple sub-queries
│   ├── retriever.py              # Retrieves documents for queries
│   └── rag_fusion.py             # Ranks results using Reciprocal Rank Fusion
│
├── services/
│   ├── embeddings.py             # Embedding generation for documents
│   ├── chat.py                   # Chat functionality with prompt-based QA
│   └── text_extraction.py        # Text extraction from uploaded documents
│
├── main.py                       # Entry point for the application
├── requirements.txt              # List of Python dependencies
└── README.md                     # Project documentation
