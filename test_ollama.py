    
import requests

url = "http://localhost:11434/api/generate"

data = {
    "model": "qwen3:4b",
    "prompt": "Explain what RAG is in simple words.",
    "stream": False
}

response = requests.post(url, json=data)

print(response.json()["response"])
