# test
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# LangChain model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key
)

# -----------------------------
# 1. LangChain request
# -----------------------------
response = llm.invoke("Hello")

print("AI Response:", response.content)

print("\n--- Token Usage ---")
print("Input Tokens:", response.usage_metadata["input_tokens"])
print("Output Tokens:", response.usage_metadata["output_tokens"])
print("Total Tokens:", response.usage_metadata["total_tokens"])


# -----------------------------
# 2. Groq rate-limit information
# -----------------------------
client = Groq(api_key=api_key)

groq_response = client.chat.completions.with_raw_response.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print("\n--- Groq Rate Limits ---")

headers = groq_response.headers

print("Remaining Requests:", headers.get("x-ratelimit-remaining-requests"))
print("Remaining Tokens:", headers.get("x-ratelimit-remaining-tokens"))

print("Request Limit:", headers.get("x-ratelimit-limit-requests"))
print("Token Limit:", headers.get("x-ratelimit-limit-tokens"))

print("Request Reset:", headers.get("x-ratelimit-reset-requests"))
print("Token Reset:", headers.get("x-ratelimit-reset-tokens"))