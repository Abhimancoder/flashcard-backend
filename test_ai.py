import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("Token nahi mila! Check karo ki .env file me HF_TOKEN set hai ya nahi.")

client = InferenceClient(api_key=token)

print("Connecting to Hugging Face model...")

response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": ""
        "who is abhiman"}
    ],
    max_tokens=60,
    temperature=0.3
)

print("\nModel Response:")
print(response.choices[0].message.content)