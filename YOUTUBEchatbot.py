from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
import os

load_dotenv()
#  INDEXING (document ingestion)

video_id = "LPZh9BOjkQs" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)


except TranscriptsDisabled:
    print("No captions available for this video.")


# TEXT SPLITTING

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# CONVERTING TO EMBEDDING AND STORING IT IN FAISS VECTOR DB
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY_2")
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

#   RETRIEVER
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
print(retriever.invoke("what is RLHF"))

# AUGEMENTATION

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash",temperature=0.7 , google_api_key=os.getenv("GOOGLE_API_KEY_2"))

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "How does training in LLM look like"
retrieved_docs    = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})

