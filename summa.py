from google import genai

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
sys.path.append("C:/Users/anand/law_assistant_app/backendfolder/indicnlp/indic_nlp_library-master")
sys.path.append("C:/Users/anand/law_assistant_app/backendfolder/rank_bm25")
os.environ['INDIC_RESOURCES_PATH'] = "C:/Users/anand/law_assistant_app/backendfolder/indicnlp/indic_nlp_library-master"

# ==== Load models and data ====
with open("C:/Users/anand/law_assistant_app/backendfolder/bm25_model.pkl", 'rb') as f:
    bm25, law_objects = pickle.load(f)

normalizer = IndicNormalizerFactory().get_normalizer("ta")
model = SentenceTransformer("intfloat/multilingual-e5-base")


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

def faiss_rerank(query_embedding, doc_embeddings, top_laws,query, top_k=3):
    embedding_dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(doc_embeddings)

    D, I = index.search(np.array([query_embedding]), top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        law = top_laws[idx]
        results.append((law, score))
    client = genai.Client(api_key="AIzaSyDOWF3YIg90xj5PFl7hUeRWSL8GokuOUFQ")

    

    law_blocks = ""
    for i, (law, score) in enumerate(results, start=1):
        law_text = law["law"]
        law_blocks += f"சட்டம் {i}:\n{law_text.strip()}\n\n"

    prompt = f"""பின்வரும் 3 சட்டங்களை வைத்து கீழ்க்கண்ட கேள்விக்கு ஒரு தெளிவான, சுருக்கமான மற்றும் நடைமுறையிலான விளக்கம் அளி.சட்ட எண்ணை குறிப்பிட்டு கூற வேண்டாம்.ஒரே பதிலாக கூறவும்

    கேள்வி: {query}

    {law_blocks}
    விளக்கம்:"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    result=response.text
    
    return result


