from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# 12 documents spanning medicine, architecture, finance, and literature
docs = [
    Document(page_content="Antibiotics inhibit bacterial cell wall synthesis or protein production to stop infection.", metadata={"topic": "medicine"}),
    Document(page_content="Vaccines introduce antigens to train the immune system to recognise and neutralise pathogens.", metadata={"topic": "medicine"}),
    Document(page_content="MRI scanners use magnetic fields and radio waves to produce detailed images of soft tissue.", metadata={"topic": "medicine"}),
    Document(page_content="Blood pressure is measured in millimetres of mercury (mmHg) and expressed as systolic over diastolic.", metadata={"topic": "medicine"}),
    Document(page_content="The Pantheon in Rome was built around 125 AD and still has the world's largest unreinforced concrete dome.", metadata={"topic": "architecture"}),
    Document(page_content="Gothic cathedrals use flying buttresses to transfer roof weight outward, enabling tall stained glass windows.", metadata={"topic": "architecture"}),
    Document(page_content="The Bauhaus movement combined fine arts and functional design, influencing modern architecture and typography.", metadata={"topic": "architecture"}),
    Document(page_content="Compound interest calculates returns on both the initial principal and previously earned interest.", metadata={"topic": "finance"}),
    Document(page_content="A stock represents partial ownership in a company and entitles the holder to a share of its profits.", metadata={"topic": "finance"}),
    Document(page_content="Diversification reduces portfolio risk by spreading investments across different asset classes.", metadata={"topic": "finance"}),
    Document(page_content="Shakespeare wrote 37 plays and 154 sonnets, exploring themes of power, love, and betrayal.", metadata={"topic": "literature"}),
    Document(page_content="The novel Don Quixote by Cervantes, published in 1605, is often considered the first modern novel.", metadata={"topic": "literature"}),
]