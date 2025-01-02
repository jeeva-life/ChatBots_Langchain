from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAI
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

# Define the function to generate LinkedIn post from summary
def generate_linkedin_post(summary: str) -> str:
    # Format the prompt with the summary
    formatted_prompt = linkedin_post_prompt.format(summary=summary)
    
    # Use ChatOpenAI model to generate the post
    linkedin_post = chat_openai(formatted_prompt)
    
    return linkedin_post


linkedin_post_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""
    Based on the following summary, create a LinkedIn post that is professional, engaging, and concise. The post should highlight the key points and attract attention from professionals in the field. Keep the tone friendly yet formal, and encourage engagement through a call to action.
    
    Summary:
    {summary}

    LinkedIn Post:
    """
)


# Define the prompt template
summary_prompt = PromptTemplate(
    input_variables=["input_text"],
    template="""
    You are an AI assistant tasked with summarizing the input text. Your job is to read the entire text carefully and then generate a brief, clear, and concise summary that captures the main points. Your summary should be under 200 words and include all the essential information.

    Input Text:
    {input_text}

    Your Summary:
    """
)

# Example usage of the prompt template
input_text = "The quick brown fox jumps over the lazy dog. It is a sunny day, and the fox is excited to explore the forest. The dog, however, is uninterested in the fox's antics and enjoys lounging in the sun. After some time, the fox notices a butterfly fluttering by and decides to chase it."

# Fill the prompt template with the input text
formatted_prompt = summary_prompt.format(input_text=input_text)

# Create a ChatOpenAI instance with parameters
chat_openai = ChatOpenAI(
    model="gpt-3.5-turbo",      # Model version
    temperature=0.7,            # Controls randomness (0.0 to 1.0)
    max_tokens=300,             # Maximum number of tokens to generate
    top_p=1.0,                  # Controls nucleus sampling (0.0 to 1.0)
    frequency_penalty=0.0,      # Penalizes repetition (range: -2.0 to 2.0)
    presence_penalty=0.0,       # Penalizes new topics (range: -2.0 to 2.0)
    stop=["\n"],                # Stops generation when this token is encountered
    api_key=api_key,            # Your OpenAI API key (ensure it's secure)
)




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

    # Load the SummarizeChain with Map-Reduce strategy
    summarize_chain = load_summarize_chain(
        llm=chat_openai,
        chain_type="map_reduce",  # Using map-reduce strategy for summarization
        verbose=True,
    )

    summary = summarize_chain.run(chunks)
    

    linkedin_post = generate_linkedin_post(summary)
    print(linkedin_post)



