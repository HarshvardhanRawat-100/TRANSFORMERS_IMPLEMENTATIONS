from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os 

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768,  # Suggested: 768, 1536, or 3072 (default)
)

query = "what is motlbot and how can that help an ai engineer"
vector = embeddings.embed_query(text=query)

#print(len(vector))

loader = PyPDFLoader(
    "/Users/harshvardhanrawat/Desktop/TRANSFORMER/ADVANCE_RAG/embeddings/Openclaw_Research_Report.pdf"
)

docs = loader.load()

print("no. of docs : " , len(docs))
#chunking


chunker = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunk = chunker.split_documents(docs)
print("no. of chunks : " , len(chunk))

text_doc = [docs.page_content for docs in chunk]
doc_embeddings = embeddings.embed_documents(texts=text_doc)

print("len of doc embeddings : " , len(doc_embeddings))
