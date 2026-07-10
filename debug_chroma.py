from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./pharma_knowledge_base",
    embedding_function=embedder
)

#Check what framework names are actually stored
collection = vectorstore._collection
results = collection.get()
frameworks = set([m['framework'] for m in results['metadatas']])
print("Frameworks stored in ChromaDB:")
for f in frameworks:
    print(f" '{f}'")