from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
import os

load_dotenv()

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

#tool1 simulates checing release status
@tool
def get_release_status(release_id:str)->str:
    """Returns the current status of release givem its release ID."""
    statuses={
        "REL-001":"QA review in progress-80% complete",
        "REL-002":"CAB approved-Ready for deployment",
        "REL-003":"Blocked-Critical bug found in system"
    }
    return statuses.get(release_id,F"Release {release_id} not found in system")

#Tool2-Simulated risk calcuulation
@tool
def Calculate_risk_score(testing_percent:int , qa_signed_off:bool,open_bugs:int)->str:
    """calculate release risk score based on testing completion,QA sign-off and open bugs."""
    score=0
    if testing_percent<80:
        score+=3
    if not qa_signed_off:
        score+=3
    if open_bugs>0:
        score+=open_bugs

        if score==0:
            return"Risk Score: LOW-Safe to proceed"
        elif score<=3:
            return "Risk Score:MEDIUM-Proceed with caution"
        else:
            return f"Risk score: HIGH(score={score}_-Do not deploy"

#Bind tools to LLM
llm_with_tools=llm.bind_tools([get_release_status,Calculate_risk_score])

#Test 1- Ask about release
print("===Test1:release status check===")
messages=[HumanMessage(content="whatis the status of the release REL-002?")]
response=llm_with_tools.invoke(messages)
print(f"LLM Decision: {response.tool_calls}")

print()

#test 2-Ask for risk calculation
print("===Test2:Risk Calculation===")
messages2=[HumanMessage(content="Calculate risk for a release with 90% testing completed, QA signed off and 0 open bugs")]
response2=llm_with_tools.invoke(messages2)
print(f"LLM decision:{response2.tool_calls}")

#Complete the tool execution loop
print()
print("===Test3:Full Agrent Loop===")

#step 1- user message
from langchain_core.messages import SystemMessage

messages3=[
    SystemMessage(content="You are a release management assisttant. Only use the tools provided to you.Do not use any other tools."),
    HumanMessage(content="what is the status of te release REL-003)")
]

#step2-LLM decides which tool to call
response3=llm_with_tools.invoke(messages3)

#step3 -Execute the tol the llm choose
tool_call= response3.tool_calls[0]
tool_name=tool_call['name']
tool_args=tool_call['args']

print(f"Agent choose tool:{tool_name}")
print(f"With arguments:{tool_args}")
      
#Step4- Actually run the tool
tool_result=get_release_status.invoke(tool_args)
print(f"Tool returned:{tool_result}")

#step5 Feed result back to LLM for final response
from langchain_core.messages import ToolMessage
messages3.append(response3)
messages3.append(ToolMessage(
    content=tool_result,
    tool_call_id=tool_call['id']
))

from langchain_groq import ChatGroq

llm_final=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

final_response = llm_final.invoke(messages3)
print(f"Final Agent Response: {final_response.content}")

