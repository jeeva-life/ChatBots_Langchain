import streamlit as st
import asyncio
from services.chat import setup_qa, conversational_chat
from services.embeddings import get_doc_embeddings, get_blog_embeddings
from services.text_extraction import extract_text_from_url
from services import get_answer_with_rag_fusion
import logging

# Set up logging configuration
logging.basicConfig(level=logging.ERROR,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("app_errors.log"), logging.StreamHandler()])

async def main():
    try:
        st.set_page_config(
            page_title="PDF_CHAT",
            layout="centered",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://docs.streamlit.io/',
                'Report a Bug': 'https://github.com/streamlit/streamlit/issues',
                'About': 'This is a PDF Chat app built with Streamlit.'
            }
        )

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        st.title("PDF_CHAT")
        option = st.selectbox("Select option", ("PDF", "Blog", "Database"))

        if option == "PDF":
            uploaded_file = st.file_uploader("Choose a file", type="PDF")
            if uploaded_file:
                with st.spinner("Processing..."):
                    try:
                        vectors = await get_doc_embeddings(uploaded_file)
                        qa = setup_qa(vectors)
                        st.session_state["ready"] = True
                    except Exception as e:
                        logging.error(f"Error processing PDF file: {e}")
                        st.error(f"Error processing PDF: {str(e)}")

        elif option == "Blog":
            url = st.text_input("Enter the URL of the blog")
            if url:
                with st.spinner("Processing..."):
                    try:
                        content = extract_text_from_url(url)
                        vectors = await get_blog_embeddings(content)
                        qa = setup_qa(vectors)
                        st.session_state["ready"] = True
                    except Exception as e:
                        logging.error(f"Error processing blog URL: {e}")
                        st.error(f"Error processing Blog: {str(e)}")

        if st.session_state.get('ready', False):
            user_input = st.text_input("Query", placeholder="e.g., Summarize the document")
            if user_input:
                try:
                    response = await conversational_chat(qa, user_input, st.session_state['history'])
                    st.write(response)
                except Exception as e:
                    logging.error(f"Error during conversational chat: {e}")
                    st.error(f"Error generating response: {str(e)}")

    except Exception as general_error:
        st.error(f"An unexpected error occurred: {str(general_error)}")


if __name__ == "__main__":
    asyncio.run(main())
