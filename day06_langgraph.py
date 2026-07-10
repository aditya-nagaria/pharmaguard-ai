from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import os

load_dotenv()

#step 1 - Define State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Step 2- Define tools
@tool
def get_release_status(release_id:str)->str:
    """Returns the current status of a release given its release ID."""
    statuses={
        "REL-001":"QA review in progress-80% complete",
        "REL-002":"CAB approved-ready for deployment",
        "REL-003":"Blocked- critical bug found in testing"
    }
    return statuses.get(release_id,f"Release{release_id} not found")

@tool
def calculate_risk_score(testing_percent:int, qa_signed_off:bool, open_bugs:int)->str:
    """Calculates release risk score based on testing completion, AQ sign-off and open bugs."""
    score=0
    if testing_percent<80:
        score+=3
    if not qa_signed_off:
        score+=3
    if open_bugs>0:
        score +=open_bugs
    if score==0:
        return "risk Score: LOW- Safe to proceed"
    elif score <=3:
        return "Risk score: MEDIUM- Proced with caution"
    else:
        return f"Risk Score: HIGH (score={score})- Do not deploy" 
    
tools=[get_release_status, calculate_risk_score]

#step3- Initialise the LLM with tools
llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools=llm.bind_tools(tools)

#step4 - Define nodes
def agent_node(state:AgentState):
    """The reasoning node-LLM decides what to do next."""
    system=SystemMessage(content="""You are a release management assistant.
    You have access to tools. Use them exactly as defined.
    Only call one tool at a time.
    Do not call any tools not in your toolset.""")
    messages=[system]+state["messages"]
    response=llm_with_tools.invoke(messages)
    return{"messages":[response]}

#step 5- Define routing logic
def should_continue(state:AgentState):
    """Decided whether to call a tool or end."""
    last_message=state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

#Step 6- Build the graph
tool_node=ToolNode(tools)

graph=StateGraph(AgentState)
graph.add_node("agent",agent_node)
graph.add_node("tools",tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools":"tools",
        "end":END
    }
)

graph.add_edge("tools","agent")

app=graph.compile()

#step7- Run the agent
print("===Test 1: Release Status===")
result = app.invoke({
    "messages":[
        SystemMessage(content="You are a release management assistant. Use tools to answer questions accurately."),
        HumanMessage(content="what is the status of release REL-001?")
    ]
})
print(result["messages"][-1].content)

print()

print("===Test2:Risk Assessment===")
result2=app.invoke({
    "messages":[
        SystemMessage(content="You are a release management assistant. Use tools to answer questions accurately."),
        HumanMessage(content="Calculate risk for a release with 70% testing, QA not signed off, and 2 open bugs")
    ]
})
print(result2["messages"][-1].content)
