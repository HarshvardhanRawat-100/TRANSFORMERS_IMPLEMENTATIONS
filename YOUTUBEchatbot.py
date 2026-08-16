from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI

#  INDEXING (document ingestion)

video_id = "LPZh9BOjkQs" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")


# TEXT SPLITTING

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
