from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
import os 

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768
)
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY_4"))

# Dummy documents covering different topics, each with a mix of relevant and tangential info
docs = [
    Document(
        page_content=(
            "Artificial intelligence has made remarkable strides in natural language processing, "
            "with large language models now capable of generating human-quality text and code. "
            "Computer vision systems can identify objects in images with superhuman accuracy, "
            "powering applications from autonomous vehicles to medical imaging diagnostics. "
            "However, the rapid advancement of AI has raised significant ethical concerns about "
            "job displacement, algorithmic bias, and the concentration of power among a few tech companies."
        ),
        metadata={"topic": "artificial_intelligence"},
    ),
    Document(
        page_content=(
            "Global temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial "
            "times, driven primarily by the burning of fossil fuels. The melting of polar ice caps has "
            "accelerated, contributing to rising sea levels that threaten coastal communities worldwide. "
            "Renewable energy adoption is growing rapidly, with solar and wind power becoming cheaper "
            "than coal in many regions. Governments are implementing carbon pricing mechanisms and "
            "investing in green infrastructure to meet Paris Agreement targets."
        ),
        metadata={"topic": "climate_change"},
    ),
    Document(
        page_content=(
            "NASA's Artemis program aims to return humans to the Moon by the mid-2020s, establishing "
            "a sustainable presence as a stepping stone to Mars. Private companies like SpaceX are "
            "developing reusable rocket technology that has dramatically reduced launch costs. "
            "The James Webb Space Telescope has captured unprecedented images of distant galaxies, "
            "revealing new insights about the early universe. Asteroid mining is being explored as a "
            "potential source of rare minerals needed for electronics manufacturing."
        ),
        metadata={"topic": "space_exploration"},
    ),
    Document(
        page_content=(
            "CRISPR gene editing technology has revolutionized medical genomics, enabling precise "
            "modifications to DNA sequences that were previously impossible. Researchers are using "
            "genomic data to develop personalized medicine approaches, tailoring treatments based on "
            "an individual's genetic profile. Recent breakthroughs in mRNA technology, accelerated by "
            "COVID-19 vaccine development, are now being applied to cancer immunotherapy and rare "
            "genetic disorders. Hospital information systems are increasingly integrating genomic data "
            "to support clinical decision-making at the point of care."
        ),
        metadata={"topic": "medicine"},
    ),
    Document(
        page_content=(
            "The global economy is navigating a period of high inflation driven by supply chain "
            "disruptions, energy price volatility, and post-pandemic demand surges. Central banks "
            "worldwide have raised interest rates aggressively to combat inflation, impacting housing "
            "markets and consumer spending. Cryptocurrency regulation is becoming a priority for "
            "financial authorities, with the EU's MiCA framework setting a global precedent. "
            "Trade tensions between major economies continue to reshape global supply chains, "
            "pushing companies toward nearshoring and diversification strategies."
        ),
        metadata={"topic": "economics"},
    ),
    Document(
        page_content=(
            "Quantum computing has reached a critical milestone with several companies demonstrating "
            "quantum advantage on specific computational tasks. Error correction remains the biggest "
            "challenge, as current quantum processors are highly susceptible to noise and decoherence. "
            "Quantum simulation of molecular structures could transform drug discovery by accurately "
            "modeling protein folding and chemical interactions. Major tech companies and governments "
            "are investing billions in quantum research, viewing it as essential for national security "
            "and economic competitiveness."
        ),
        metadata={"topic": "quantum_computing"},
    ),
]

print(f"Created {len(docs)} documents")