import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# 15 documents: the first 3 deep learning docs are intentionally near-identical —
# all describe gradient descent as the core optimisation technique, with minor phrasing variation.
# The next 3 deep learning docs cover distinct regularisation/training concepts.
# At lambda_mult=1.0 (pure relevance), a query about deep learning training returns
# all 3 near-identical gradient descent docs, showing the redundancy problem.
# As lambda_mult decreases, MMR penalises already-selected similar docs and
# picks one gradient descent doc + Dropout + Batch Norm + LR Scheduler instead.
docs = [
    Document(page_content="Training a deep learning model involves iteratively adjusting weights using gradient descent to minimise the loss.", metadata={"topic": "deep learning"}),
    Document(page_content="Deep learning models are optimised through gradient descent, which updates weights in the direction that reduces the training loss.", metadata={"topic": "deep learning"}),
    Document(page_content="Gradient descent is the core optimisation technique in deep learning, guiding weight updates based on computed gradients of the loss.", metadata={"topic": "deep learning"}),
    Document(page_content="Dropout randomly disables a fraction of neurons during training to prevent overfitting in deep networks.", metadata={"topic": "deep learning"}),
    Document(page_content="Batch normalisation stabilises training by normalising layer inputs, which allows the use of higher learning rates.", metadata={"topic": "deep learning"}),
    Document(page_content="Learning rate schedulers dynamically adjust the learning rate during training to improve convergence and avoid overshooting.", metadata={"topic": "deep learning"}),
    Document(page_content="Arctic sea ice has declined by about 13% per decade since satellite measurements began in 1979.", metadata={"topic": "climate"}),
    Document(page_content="Carbon capture technology removes CO2 from the atmosphere and stores it underground.", metadata={"topic": "climate"}),
    Document(page_content="The permafrost in Siberia contains vast amounts of methane that could be released as it thaws.", metadata={"topic": "climate"}),
    Document(page_content="The Renaissance was a cultural movement in Europe from the 14th to 17th century that revived classical art.", metadata={"topic": "art"}),
    Document(page_content="Impressionism emerged in 19th-century France, focusing on light, colour, and everyday subjects.", metadata={"topic": "art"}),
    Document(page_content="Abstract expressionism prioritises spontaneous, automatic, and subconscious creation.", metadata={"topic": "art"}),
    Document(page_content="Common law systems derive legal principles from judicial precedent rather than written codes.", metadata={"topic": "law"}),
    Document(page_content="The presumption of innocence requires the prosecution to prove guilt beyond reasonable doubt.", metadata={"topic": "law"}),
    Document(page_content="Intellectual property law protects creations of the mind, including patents, trademarks, and copyrights.", metadata={"topic": "law"}),
]
