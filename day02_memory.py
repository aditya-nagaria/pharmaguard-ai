from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client=Groq(api_key=os.getenv("Groq_API_KEY"))

messages = [
    {"role":"system","content":"You are an expert in pharmaceutical IT release management. Answer precisely and technically"}
]

print("Chat Started. Type 'quit' to exit")
print("-"*40)

while True:
    #Get input from user
    user_input=input("You: ")

    #Exit condition
    if user_input.lower()=="quit":
        print("chat ended.")
        break

    #Add user to message history
    messages.append({"role":"user","content":user_input})

    #call the API with Full history
    response=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    #Extract assistane response
    assistant_reply=response.choices[0].message.content

    #Add assistant response to history
    messages.append({"role":"assistant","content":assistant_reply})

    #print the response
    print(f"Assistant: {assistant_reply}")
    print("-"*40)
