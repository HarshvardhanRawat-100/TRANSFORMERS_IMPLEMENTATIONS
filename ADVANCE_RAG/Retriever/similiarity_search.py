import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Create a list of documents with metadata
docs = [
    Document(page_content="Rockets work by expelling gas at high speed, generating thrust through Newton's third law of motion.",
             metadata={"topic": "space"}),
    Document(page_content="The International Space Station orbits Earth at about 400 km altitude and travels at 28,000 km/h.",
             metadata={"topic": "space"}),
    Document(page_content="Spacecraft use gravitational slingshots around planets to gain speed without burning extra fuel.",
             metadata={"topic": "space"}),
    Document(page_content="NASA's Voyager 1 is the farthest human-made object, now over 23 billion km from the Sun.",
             metadata={"topic": "space"}),
    Document(page_content="Solar sails use radiation pressure from sunlight to slowly propel spacecraft without fuel.",
             metadata={"topic": "space"}),
    Document(page_content="DNA is a double-helix molecule that carries the genetic instructions for all living organisms.",
             metadata={"topic": "biology"}),
    Document(page_content="Photosynthesis allows plants to convert sunlight, water, and CO2 into glucose and oxygen.",
             metadata={"topic": "biology"}),
    Document(page_content="The Roman Empire at its peak covered over 5 million square kilometers across three continents.",
             metadata={"topic": "history"}),
    Document(page_content="The printing press, invented by Gutenberg around 1440, revolutionised the spread of knowledge.",
             metadata={"topic": "history"}),
    Document(page_content="The Amazon River discharges more freshwater into the ocean than any other river on Earth.",
             metadata={"topic": "geography"}),
    Document(page_content="The Sahara Desert spans about 9.2 million square kilometers across northern Africa.",
             metadata={"topic": "geography"}),
    Document(page_content="Mitochondria generate ATP through cellular respiration, powering nearly all cellular processes.",
             metadata={"topic": "biology"}),
]

for i , doc in enumerate(docs,1) :
    print(f"Doc no. : {i} | Topic : {doc.metadata['topic']} ")
    print(f"Content : {doc.page_content}\n")

embed = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="similarity_search_demo",
)

retriever = vectorstore.as_retriever(
    search_type = 'similarity',
    search_kwargs = {"k":2}
)

