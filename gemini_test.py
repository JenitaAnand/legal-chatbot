import google.generativeai as genai

genai.configure(api_key="AIzaSyDOWF3YIg90xj5PFl7hUeRWSL8GokuOUFQ")  # your key

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Hi darling, how are you?")

print("🟢 Gemini Response:", response.text)
