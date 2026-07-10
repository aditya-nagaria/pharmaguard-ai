from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

#Step 1- Initialise embedding model
embedder = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#Step 2 - Embed some pharma regulatory text
texts = [
    "Electronic signatures must be unique to one individual and not resused by anyone else",
    "AUdit trails must records date and time of operator entries and actions.",
    "Software must be validated to ensure accuracy and reliablity of results.",
    "IQ verifies that equipment is installed correctly according to manufacturer specfications",
    "OQ verifies that equipment operates withing defined parameters."
]

print("===Generating Embeddings===")
embeddings = embedder.embed_documents(texts)

print(f"Number of exts embedded: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")
print(f"First 5 values of embedding 1: {embeddings [0][:5]}")

print()

#Step 3 - Embed a query and compare
print("===Query Embedding===")
query = "what are the requirements for elctronic signatures?"
query_embedding = embedder.embed_query(query)
print(f"Query embedding dimensions:{len(query_embedding)}")
print(f"First 5 values:{query_embedding[:5]}")

print()

#step 4 Manual similarity check
print("===Similarity Check===")
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

for i, text in enumerate(texts):
    similarity = cosine_similarity(query_embedding, embeddings[i])
    print(f"Score {similarity: 4f} | {text[:60]}...")
