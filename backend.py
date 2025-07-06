from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from google import genai
import google.generativeai as genai
# ==== Your current imports ====
import sys
import os
import pickle
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ==== Paths ====
sys.path.append("./indic_nlp_library-master")
sys.path.append("./rank_bm25")
os.environ['INDIC_RESOURCES_PATH'] = "./indic_nlp_library-master"

# ==== Load models and data ====
with open("./bm25_model.pkl", 'rb') as f:
    bm25, law_objects = pickle.load(f)


normalizer = IndicNormalizerFactory().get_normalizer("ta")
model = SentenceTransformer("intfloat/multilingual-e5-base")

# ==== Configure Gemini API ====
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ==== FastAPI setup ====
app = FastAPI()

# Allow frontend access
allowed_origins = [
    "http://localhost",
    "http://10.0.2.2",  # Android emulator
    "http://192.168.1.10",  # Your local IP
    "capacitor://localhost",  # Mobile hybrid
    "https://yourdomain.com"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Pydantic model ====
class QueryRequest(BaseModel):
    text: str

# ==== Core logic ====
def get_top_bm25_indices(query, bm25, top_k=10):
    query_norm = normalizer.normalize(query)
    query_tokens = indic_tokenize.trivial_tokenize(query_norm, lang='ta')
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return top_indices, scores

def encode_laws_for_faiss(query, top_laws):
    query_encoded = "query: " + query
    texts = ["passage: " + law["content"] for law in top_laws]
    doc_embeddings = model.encode(texts, normalize_embeddings=True)
    query_embedding = model.encode(query_encoded, normalize_embeddings=True)
    return doc_embeddings, query_embedding



genai.configure(api_key="AIzaSyDOWF3YIg90xj5PFl7hUeRWSL8GokuOUFQ")

def faiss_rerank(query_embedding, doc_embeddings, top_laws, query, top_k=3):
    try:
        index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        index.add(doc_embeddings)

        D, I = index.search(np.array([query_embedding]), top_k)

        results = [(top_laws[idx], score) for idx, score in zip(I[0], D[0])]

        law_blocks = ""
        for i, (law, _) in enumerate(results, start=1):
            law_blocks += f"சட்டம் {i}:\n{law['law'].strip()}\n\n"

        prompt = f"""பின்வரும் 3 சட்டங்களை வைத்து கீழ்க்கண்ட கேள்விக்கு ஒரு தெளிவான, சுருக்கமான மற்றும் நடைமுறையிலான விளக்கம் அளி. சட்ட எண்ணை குறிப்பிடாதே. ஒரே பதிலாக கூறவும்:

        கேள்வி: {query}

        {law_blocks}
        விளக்கம்:"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        print("❌ Gemini API Error:", str(e))
        return "பதில் பெறுவதில் பிழை"
 # Gemini returns only a string now

# ==== API endpoint ====

@app.post("/api/ask")
async def ask_query(request: QueryRequest):
    query = request.text
    top_indices, scores = get_top_bm25_indices(query, bm25)
    top_laws = [law_objects[idx] for idx in top_indices]
    doc_embeddings, query_embedding = encode_laws_for_faiss(query, top_laws)
    results = faiss_rerank(query_embedding, doc_embeddings, top_laws, query)

    # Handle Gemini's string response
    if isinstance(results, str):
        top_response = results
        relevant_laws = [law["law"] for law in top_laws]
    else:
        top_response = results[0][0]["law"] if results else "மன்னிக்கவும், தகவல் கிடைக்கவில்லை."
        relevant_laws = [r[0]["law"] for r in results]

    return {
        "response": top_response,
        "relevant_laws": relevant_laws
    }

@app.get("/ping")
async def ping():
    return {"status": "pong"}

# ==== Run app ====
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

    










