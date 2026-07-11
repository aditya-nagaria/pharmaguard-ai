from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import json

load_dotenv()

#============LLM==============
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

#---Embedder + Vector Store
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./pharma_knowledge_base",
    embedding_function=embedder
)

#----State-----------------
class PharmaGuardState(TypedDict):
    task_description: str
    task_type: str
    task_scope: str
    applicable_frameworks: list
    retrieved_requirements: list
    deviations: list
    risk_level: str
    compliance_brief: dict
    messages: Annotated[list, add_messages]

#-------Agent1: INTAKE AGENT--------------------------
def intake_agent(state: PharmaGuardState):
    print(">>Intake Agent running...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a pharma IT intake specialist.
         Classify the task and respond ONLY in valid JSON with these exact keys:
         - task_type: one of 'validation', 'release', 'audit', 'documentaion'
         -task_scope: one sentence describing what is being done
         No extra text, no markdown, just JSON."""),
         ("human","Task: {task_description}")
    ])

    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({
        "task_description": state["task_description"]
    })

    return {
        "task_type": result["task_type"],
        "task_scope": result["task_scope"],
        "messages": [HumanMessage(content=f"Intake complete: {result}")]
    }

#Agent2: Framework selector------
def framework_selector(state: PharmaGuardState):
    print(">>Framework Selector running...")

    prompt = ChatPromptTemplate.from_messages([
        ("system","""You are a pharma regulatory expert.
         Based on the task type and scope, select applicable frameworks.
         Avaiable frameworks: GxP, 21 CFR Part 11, GAMP 5, ISO 13485, IEC 62304. Audit Readiness)
         Respond ONLY in valid JSON with this exact key:
         - applicable_frameworks: list of applicable framework names
         No extra text, no markdown, just JSON."""),
         ("human","""Task Type: {task_type}
Task Scope: {task_scope}""")
    ])

    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({
        "task_type": state["task_type"],
        "task_scope": state["task_scope"]
    })

    return {
        "applicable_frameworks": result["applicable_frameworks"],
        "messages": [HumanMessage(content=f"Frameworks selected: {result['applicable_frameworks']}")]
    }

#-------------Agent 3- Requirements Agent-------------
def requirements_agent(state: PharmaGuardState):
    print(">>Requirements agent running")

    frameworks = state["applicable_frameworks"]
    task_scope = state["task_scope"]
    all_requirements = []

    for framework in frameworks:
        #Query ChromaDB for each framework
        query = f"{task_scope} requirements for {framework}"
        docs = vectorstore.similarity_search(
            query,
            k=2,
            filter={"framework": framework}
        )

        for doc in docs:
            all_requirements.append({
                "framework": framework,
                "requirements": doc.page_content,
                "category": doc.metadata.get("category","general")
            })

    return {
        "retrived_requirements": all_requirements,
        "messages": [HumanMessage(content=f"Retrieved {len(all_requirements)} requirements")]
    } 

def requirements_agent(state: PharmaGuardState):
    print(">> Requirements Agent running...")

    frameworks = state["applicable_frameworks"]
    task_scope = state["task_scope"]
    all_requirements = []

    
    for framework in frameworks:
        query = f"{task_scope} requirements for {framework}"
        
        docs = vectorstore.similarity_search(
            query,
            k=2,
            filter={"framework": framework}
        )

    
        for doc in docs:
            all_requirements.append({
               "framework": framework,
               "requirement": doc.page_content,
               "category": doc.metadata.get("category", "general") 
            })
    return {
        "retrieved_requirements": all_requirements,
        "messages": [HumanMessage(content=f"Retrived {len(all_requirements)} requirements")]
    }

#-----Agent 4- Deviation checker---------
def deviation_checker(state: PharmaGuardState):
    print(">> Deviatin Checker running..")

    requirements_text = "\n".join([
        f"[{r['framework']}] {r['requirement']}"
        for r in state["retrieved_requirements"]
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system","""You are a pharma compliance auditor.
         Compare the task description against the regulatory requirements.
        Identify which requirements are NOT clearly addressed by the task description.
        Respond ONLY in valid JSON with these exact keys:
         -deviations: list of strings, each descrinbing one gap or missing requirements
         -risk_level: 'low','medium', or 'high'
         Risk is 'high' if 2 or more critical deviations exists, 'medium' if 1, 'low' if none.
         No extra text, no markdown, just JSON."""),
         ("human","""Task Description:{task_description}

Requirements:
          {requirements}""")
    ])

    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({
        "task_description": state["task_description"],
        "requirements": requirements_text
    })

    return {
        "deviations": result["deviations"],
        "risk_level": result["risk_level"],
        "messages": [HumanMessage(content=f"Deviation check complete: {result['risk_level']} risk")]
    }

#---- Agent 5: Report Generator-------------
def report_generator(state: PharmaGuardState):
    print(">> Report Generator running...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a pharma compliance report writer.
        Synthesize the analysis into a final compliance brief.
        Respond ONLY in valid JSON with these exact keys:
        - summary: 2-3 sentence executive summary
        - dos: (exact key name "dos", plural) list of 3-5 specific actions required before proceeding
        - donts: list of 2-3 things to avoid
        - recommendation: 'approve', 'delay', or 'reject'
        No extra text, no markdown, just JSON."""),
        ("human", """Task Type: {task_type}
Task Scope: {task_scope}
Applicable Frameworks: {frameworks}
Risk Level: {risk_level}
Deviations: {deviations}""")
    ])

    chain = prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({
            "task_type": state["task_type"],
            "task_scope": state["task_scope"],
            "frameworks": ", ".join(state["applicable_frameworks"]),
            "risk_level": state["risk_level"],
            "deviations": "; ".join(state["deviations"])
        })
    except Exception as e:
        print(f"Warning: JSON parsing failed, retrying...")
        result = chain.invoke({
            "task_type": state["task_type"],
            "task_scope": state["task_scope"],
            "frameworks": ", ".join(state["applicable_frameworks"]),
            "risk_level": state["risk_level"],
            "deviations": "; ".join(state["deviations"])
        })

    return {
        "compliance_brief": result,
        "messages": [HumanMessage(content="Compliance brief generated")]
    }
#-------Build graph-------------
graph = StateGraph(PharmaGuardState)

graph.add_node("intake", intake_agent)
graph.add_node("framework_selector", framework_selector)
graph.add_node("requirements", requirements_agent)
graph.add_node("deviation_checker", deviation_checker)
graph.add_node("report_generator", report_generator)

graph.set_entry_point("intake")
graph.add_edge("intake", "framework_selector")
graph.add_edge("framework_selector", "requirements")
graph.add_edge("requirements", "deviation_checker")
graph.add_edge("deviation_checker", "report_generator")
graph.add_edge("report_generator", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

#Run===============
config = {"configurable": {"thread_id": " pharma_session_001"}}

print("=" * 50)
print("PHARMAGUARD AI - INTAKE + FRAMEWORK SELECTION")
print("=" * 50)

#-------- INTERACTIVE CLI------------
def run_pharmaguard():
    print("=" * 60)
    print("PHARMAGUARD AI - Pharma Compliance Co-Pilot")
    print("+" * 60)
    print("Describe your task in plain English.")
    print("Type 'quit' to exit.")
    print()

    session_count = 0

    while True:
        task_input = input("Task > ")

        if task_input.lower() == "quit":
            print("Session ended.")
            break

        session_count +=1
        thread_id = f"pharma_session_{session_count}"
        config = {"configurable": {"thread_id": thread_id}}

        print()
        print(f">>Analyzig task {session_count}...", flush=True)
        print()

        result = app.invoke({
            "task_description": task_input,
            "task_type": "",
            "task_scope": "",
            "applicable_frameworks": [],
            "retrived_requirements": [],
            "deviations": [],
            "risk_level": "",
            "compliance_brief": {},
            "messages": []
        }, config=config)    
        
        print()
        print("-" * 60)
        print(f"TASK TYPE:      {result['task_type']}")
        print(f"SCOPE:      {result['task_scope']}")
        print(f"FRAMEWORKS: {','.join(result['applicable_frameworks'])}")
        print(f"RISK LEVEL: {result['risk_level'].upper()}")
        print("-" * 60)

        brief = result['compliance_brief']
        print(f"\nSummary:\n{brief['summary']}")

        print(f"\nDOS:")
        for item in brief.get('dos', []):
            print(f"  + {item}")

        print(f"\nDON'TS:")
        for item in brief.get('donts', []):
            print(f"    -{item}")

        print(f"\nRECOMMENDATION: {brief['recommendation'].upper()}")
        print("-" * 60)
        print()

#----ENTRY POINT--
if __name__ == "__main__":
    run_pharmaguard()