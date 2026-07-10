from dotenv import load_dotenv
from groq import Groq
import os
import json

load_dotenv()

client=Groq(api_key=os.getenv("GROQ_API_KEY"))

#function 1 Basic LLM function wrapped in a function
def ask_llm(question, system_prompt):
    response=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user","content":question}
        ]
    )
    return response.choices[0].message.content

#Function 2- LLM call that reuturns structured JSON
def analyse_release_risk(release_details):
    system_promt="""You are a Pharma IT Release manager.
    Analyse the release details and respond ONLY in JSON format with no extra text.
    JSON mush have exactly these keys:
    -risk_level: either 'low','medium', or 'high'
    -reason: one sentence explaining the risk
    -recommendation: either 'approve','delay' or 'reject'
    """

    response =client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system","content":system_promt},
            {"role":"user","content":release_details}
        ]
    )

    raw_text=response.choices[0].message.content

    try:
        parsed=json.loads(raw_text)
        return parsed
    except json.JSONDecodeError:
        print(f"Warning:LLM retuned Invalid Response")
        print(f"Raw response was:{raw_text}")
        return{
            "risk_level":"unknown",
            "reason":"could not parse LLM response",
            "recommendation":"manual review required"
        }

    parsed=json.loads(raw_text)
    return parsed

#test function 1
print("=== Testing basic LLM function===")
answer=ask_llm(
    question="what is CAB in IT release management",
    system_prompt="You are an IT release management expert. Answer in 2 lines"
)
print(answer)

print()

#Test Function 2
print("=== Testing structured output function===")
release_info="Release V2.3.1 for bio4c microservice. testing is 68% complete.No QA sign off.Deployment window is tommorow "
result=analyse_release_risk(release_info)

print(f"Risk level:{result['risk_level']}")
print(f"Reason:{result['reason']}")
print(f"Recommendation:{result['recommendation']}")