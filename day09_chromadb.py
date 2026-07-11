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
    ),
    Document(
        page_content="All Bill of Materials entries must be traceable to approved system requirements and functional specifications, with no untraceable components permitted at release.",
        metadata={"framework": "Change Control", "category": "bom_management"}
    ),
    Document(
        page_content="Development, Staging, and Production environments must each maintain a distinct, documented set of test cases with corresponding evidence and sign-off records.",
        metadata={"framework": "Change Control", "category": "environment_governance"}
    ),
    Document(
        page_content="All BOM and artifact versions must carry unambiguous version identifiers, with version history maintained and auditable.",
        metadata={"framework": "Change Control", "category": "versioning"}
    ),
    Document(
        page_content="The BOM must capture complete details of libraries, applications, and versions in alignment with relevant regulatory and cybersecurity resilience requirements including the EU Cyber Resilience Act.",
        metadata={"framework": "Change Control", "category": "bom_management"}
    ),
    Document(
        page_content="All platform and application validation activities must be performed within a controlled, qualified environment with restricted and logged access.",
        metadata={"framework": "CAB Governance", "category": "controlled_environment"}
    ),
    Document(
        page_content="Validation reports and related documentation must follow an approved template and carry electronic approval consistent with 21 CFR Part 11 signature requirements.",
        metadata={"framework": "CAB Governance", "category": "documentation"}
    ),
    Document(
        page_content="A requirements traceability matrix must be maintained and kept current, reflecting the latest approved set of requirements, test cases, and evidence.",
        metadata={"framework": "CAB Governance", "category": "traceability"}
    ),
    Document(
        page_content="All artifacts must be validated prior to promotion and stored only in qualified, access-controlled environments.",
        metadata={"framework": "CAB Governance", "category": "artifact_management"}
    ),
    Document(
        page_content="Installation Qualification IQ must verify and document that equipment and software are installed correctly per manufacturer and design specifications before further qualification proceeds.",
        metadata={"framework": "GAMP 5", "category": "iq_oq_pq"}
    ),
    Document(
        page_content="Operational Qualification OQ must demonstrate the system operates as intended across its full specified operating range, with documented test evidence for each functional requirement.",
        metadata={"framework": "GAMP 5", "category": "iq_oq_pq"}
    ),
    Document(
        page_content="Performance Qualification PQ must confirm the system performs reliably under real-world operating conditions and production-representative data volumes.",
        metadata={"framework": "GAMP 5", "category": "iq_oq_pq"}
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
