from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client=Groq(api_key=os.getenv("GROQ_API_KEY"))

response=client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role":"system","content":"You are an expert in pharmaceutical IT release management. Answer precisely and technically."},
        {"role":"user","content":"What is a Change Advisory Board in IT release management?"}
    ]
)

print(response.choices[0].message.content)