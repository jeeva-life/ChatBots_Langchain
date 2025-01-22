from langchain_openai import ChatOpenAI
from prompts.query_prompts import prompt_rag_fusion

def parse_queries_output(message):
    """Parse the LLM output into a list of queries."""
    return message.content.split('\n')

llm = ChatOpenAI(temperature=0)  # Initialize the language model

def generate_queries(input_query):
    """Generate multiple queries based on the input query."""
    return prompt_rag_fusion.invoke({"question": input_query}, llm=llm, post_process=parse_queries_output)
