from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("groq_api_key")

if api_key:
    print("API key loaded succesfully")
    print(f"key starts with: {api_key[:8]}...")
else:
    print("API key Not found")
    