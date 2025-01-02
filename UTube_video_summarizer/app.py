from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
import os

api_key = os.getenv("OPENAI_API_KEY")

print(api_key)

from urllib.parse import urlparse, parse_qs

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]
    except Exception as e:
        print("Invalid URL format:", e)
        return None

# Function to fetch and print video transcript
def fetch_youtube_transcript(url):
    video_id = extract_video_id(url)
    if not video_id:
        print("Could not extract video ID from the URL.")
        return

    try:
        # Fetch transcript for the video ID
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Print the transcript
        print(f"Transcript for video ID: {video_id}")
        for entry in transcript:
            #print(f"[{entry['start']:.2f}s]: {entry['text']}")
            print(entry['text'])
        return video_id
    except Exception as e:
        print("An error occurred:", e)

def text2doc(video_id):
    try:
        # Fetch transcript for the video ID
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all transcript text into one string
        transcript_text = ' '.join([entry['text'] for entry in transcript])
        
        # Create a LangChain Document
        doc = Document(page_content=transcript_text)
        
        # Split document into chunks using a character-based splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents([doc])
        
        return chunks
    except Exception as e:
        print("An error occurred:", e)
        return None

# Example usage
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example YouTube URL
    id = fetch_youtube_transcript(youtube_url)
    chunks = text2doc(id)

    if chunks:
        # Print the first few chunks to verify
        print(f"Total Chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks for example
            print(f"Chunk {i+1}: {chunk.page_content[:300]}...")  # Preview first 300 chars of each chunk

