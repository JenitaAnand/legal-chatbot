import google.generativeai as genai

genai.configure(api_key="AIzaSyDOWF3YIg90xj5PFl7hUeRWSL8GokuOUFQ")

model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro" if you have access
response = model.generate_content(prompt)
return response.text
