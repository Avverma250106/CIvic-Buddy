import google.generativeai as genai
import os

API_KEY = "AIzaSyAge4tbii2Lo6GE46iZlHxFJrXy7FjEp-Q"   # replace with your key

genai.configure(api_key=API_KEY)

# List all available models
models = genai.list_models()

print("\n===== AVAILABLE MODELS FOR YOUR API KEY =====\n")
for m in models:
    print(f"ID: {m.name}")
    print(f"Supports: {m.supported_generation_methods}")
    print("-----------------------------------------")
