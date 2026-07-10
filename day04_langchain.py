from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import os

load_dotenv()

#Step 1 initialise the LLM
llm=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Step 2 Create a Prompt Template with Variables
prompt=ChatPromptTemplate.from_messages([
    ("system","You are an expert in {domain}. Answer precisely in 3 lines"),
    ("human","{question}")
])

#STEP3 create output parser
parser=StrOutputParser()

#step 4 build the chain using pipe operator
chain=prompt|llm|parser

#step5 Invoke the chain
print("===Test 1: Pharma IT question===")
result1=chain.invoke({
    "domain":"pharmaceutical IT release management",
    "question":"what is the purpose of Go-Nogo in releae management?"
})
print(result1)

print()

#step6-Same chain, different domain
print("===Test2:agentic AI question===")
result2=chain.invoke({
    "domain":"agentic AI systems",
    "question":"what is the difference between chat bot and AI agent?"
})
print(result2)

from langchain_core.output_parsers import JsonOutputParser

#New prompt that asks for JSON output
json_prompt=ChatPromptTemplate.from_messages([
    ("system","""You are a pharma IT Release manaager.
     Respond only in Valid JSON with these exact keys:
     -risk level:'low','medium','high'
     -reason:one sentence
     -recommendation:'approve','delay',or'reject'
     No extra text,no markdown, just JSON"""),
     ("human","{release_details}")
     ])

#swap parser to JsonOutputParser
json_parser=JsonOutputParser()

#New chain
json_chain=json_prompt|llm|json_parser

print()
print("===Test3:structured release risk analysis===")
risk_result=json_chain.invoke(  {
    "release_details":"Release version 3.1 in LIMS module.Testing is 100% complete.QA signed off."
})

print(f"Risk level:{risk_result['risk_level']}")
print(f"reason:{risk_result['reason']}")
print(f"recommendation:{risk_result['recommendation']}")

