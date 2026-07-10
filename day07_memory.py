from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import os

load_dotenv()

#state definition
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    #Tools
@tool
def get_release_status(release_id: str) -> str:
        """Returns the release status of a release given its release ID."""
        statuses={
            "REL-001":"QA Review in progrss-80% Complete",
            "REL-002":"CAB approved-ready for deployment",
            "REL-003":"Blocked-critical bug found in testing"
        }
        return statuses.get(release_id,f"Release {release_id} not found")
    
tools = [get_release_status]

#LLM
llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools=llm.bind_tools(tools)

#nodes
def agent_node(state:AgentState):
    system= SystemMessage(content="""You are a release management assitant.
    Use tools to answer questions accurately.
    Only call one tool at a time.
    Do not call any tools not in your toolset.""")
    messages=[system]+state["messages"]
    response=llm_with_tools.invoke(messages)
    return {"messages":[response]}

def should_continue(state:AgentState):
    last_message=state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

#Build Graph
tool_node = ToolNode(tools)
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools":"tools","end":END}
)
graph.add_edge("tools","agent")

#compile with memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

#Session config- same thread ID= Same conversation
config = {"configurable":{"thread_id":"release_session_001"}}

#Turn 1
print("===Turn 1===")
result1 = app.invoke({
    "messages":[HumanMessage(content="what is the status of REL-001?")]
}, config=config)
print(result1["messages"][-1].content)

print()

#Turn 2 - referencec turn 1 without repeating it
print("===Turn 2===")
result2 = app.invoke({
    "messages": [HumanMessage(content="Is that release ready for deployment?")]
}, config=config)
print(result2["messages"][-1].content)

print()

#Turn 3- check what agent remembers
print("===turn3===")
result3 = app.invoke({
    "messages":[HumanMessage(content="Summarize what we discussed so far")]
},config=config)
print(result3["messages"][-1].content)