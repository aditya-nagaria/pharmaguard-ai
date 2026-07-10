from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

#Step 1 Initialize embedder and LLM
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

#step 2 Load existing knowledge base
print("===Loading Knowledge Base ===")
vectorstore = Chroma(
    persist_directory="./pharma_knowledge_base",
    embedding_function=embedder
)
print(f"Knowledge base loaded successfully")

print()

#Step 3 Build RAG prompt Template 
rag_prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a PharmaGuard AI - an expert pharmaceutical IT compliance assistant. Answer questions based only on the provided regulatory context. If the context does not contain enough information, Say so clearly.Always cite which framework the requirement comes from."""),
    ("human","""Regulatory Context:
{context}
     
Question: {question}
     
Provide a structured compliance answer based on the context above.""")
])

#Step 4 - Build RAG function
def rag_query(question: str, framework_filter: str = None, k: int=3) -> str:
    
    #Retrive relevant documents
    if framework_filter:
        docs = vectorstore.similarity_search(
            question,
            k=k,
            filter={"framework": framework_filter}
        )
    else:
        docs=vectorstore.similarity_search(question, k=k)

    # combine retrived chunks into context
    context = "\n\n".join([
        f"[{doc.metadata['framework']}]: {doc.page_content}"
        for doc in docs
    ])

    #Build and run the chain
    chain = rag_prompt | llm | StrOutputParser()
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response

#step 5 - Test RAG queries
print("=== Test 1 : Electronic Recores query===")
answer1 = rag_query("What are the requirements for electronic signatures?")
print(answer1)

print()

print("=== Test 2: GAMP 5 Validation Query===")
answer2 = rag_query(
    "What validation is needed for custom pharma software?",
    framework_filter="Gamp 5"
)
print(answer2)

print()

print("=== Test 3 : Data Integrity Query===")
answer3 = rag_query("What are ALCOA data integrity principles in GxP?")
print(answer3)

