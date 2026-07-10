from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

load_dotenv()

#Step 1 - Initialize embedder
embedder = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#Step2- Define regulatory knowledge
regulatory_docs = [
    Document(
        page_content="Electronic signatures must be unique to one individual and not resused or reassigned to anyone else.",
        metadata={"framework":"21 CFR Part 11","category":"electronic_signatures"}
        ),
    Document(
        page_content="Audit trails must be computer generated and include date and time of operator entries an actions that create, modify, or delete elctronic records.",
        metadata={"framework":"21 CFR Part 11","category":"audit_trail"}
    ),
    Document(
        page_content="Electronic records must be protected to enable their accrate and ready retrival throughout the records retention period.",
        metadata={"framework":"21 CFR Part 11","Category":"data_integrity"}
    ),
    Document(
        page_content="All GxP computerised systems must be validated to show that they are fit for their intended use and meet regulatory requirements.",
        metadata={"framework":"GxP","category":"validation"}
    ),
    Document(
        page_content="All GxP requires that data must be attributable, legible, contemporaneous, orignal and accurate- Known as ALCOA principles.",
        metadata={"framework":"GxP","Category":"data_integrity"}
    ),
    Document(
        page_content="Gamp 5 classifies software into categories: Cateogry 1 infrastructure software, Category 3 non-configured prodcuts, Category 4 configured prodcuts, Cateogry 5 custom applications.",
        metadata={"framework":"GAMP 5","category":"software_classification"}
    ),
    Document(
        page_content="Gamp 5 category 5 custom software requires full IQ OQ PQ validation with complete traceablity from user requirements to test results.",
        metadata={"framework":"GAMP 5","category":"validation"}
    ),
    Document(
        page_content="ISO 13485 requires documented procedures for design and development including desing planning, inputs, outputs, review, verification and validation.",
        metadata={"framework":"ISO 13485","category":"design_control"}
    ),
    Document(
        page_content="IEC 62304 defined software development lifecycle for medical devide software including software development planning, requirements analysis, architecturak design, and maintenance.",
        metadata={"framework":"IEC 62304","category":"software_lifecycle"}
    ),
    Document(
        page_content="Audit readiness requires that all validation documentation, change controls, deviations, and CAPAs are complete, approved, and retrivable within 24 hours of audit request.",
        metadata={"framework":"Audit Readiness","category":"documentation"}
    )
]

#Step3- Build ChromaDB vector store
import shutil

print("===Building ChromaDB Knowledge Base===")

#Clear existing database before rebuilding
if os.path.exists("./pharma_knowledge_base"):
    shutil.rmtree("./pharma_knowledge_base")

vectorstore = Chroma.from_documents(
    documents=regulatory_docs,
    embedding=embedder,
    persist_directory="./pharma_knowledge_base"
)

#Step 4 -Query the knoewledge base
print("===Query 1:Electronic Signatures===")
query1="What are the requirements for electronic signatures and audit trails.?"
results1=vectorstore.similarity_search(query1, k=3)
for i, result in enumerate(results1):
    print(f"Result {i+1} [{result.metadata['framework']}]:{result.page_content[:80]}...")

    print()

#Step 5- Query with metadata filter
print("===Query 2: GAMP 5 Only ===")
query2="What validation is required for custom software?"
results2=vectorstore.similarity_search(
    query2,
    k=2,
    filter={"framework":"GAMP 5"}
)
for i, result in enumerate(results2):
    print(f"Result {i+1} [{result.metadata['framework']}]: {result.page_content[:80]}...")

print()

#Step 6 - Similarity search with scores
print("===Query 3:With Relevance Scores===")
query3 = "what are ALCOA data integrity principles?"
results3 = vectorstore.similarity_search_with_score(query3, k=3)
for result, score in results3:
    print(f"Score {score:.4f} | [{result.metadata['framework']}]: {result.page_content[:60]}...")
