import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("No se encontró OPENAI_API_KEY en variables de entorno.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(BASE_DIR, "vector_db")

INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(VECTOR_DIR, "metadata.pkl")

client = OpenAI(api_key=API_KEY)

if not os.path.exists(INDEX_FILE):
    raise FileNotFoundError(f"No se encontró el índice FAISS en: {INDEX_FILE}")

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(f"No se encontró metadata.pkl en: {METADATA_FILE}")

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

def embed_query(query: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding, dtype="float32")

def buscar_contexto(query: str, k: int = 5):
    vector = embed_query(query)
    vector = np.array([vector], dtype="float32")

    distances, indices = index.search(vector, k)

    resultados = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            resultados.append(metadata[idx])

    return resultados

def construir_contexto_texto(contexto):
    contexto_texto = ""

    for c in contexto:
        if c.get("tipo") == "faq":
            contexto_texto += (
                f"Tipo: Pregunta frecuente\n"
                f"Pregunta: {c.get('pregunta', '')}\n"
                f"Respuesta: {c.get('respuesta', '')}\n"
                f"URL: {c.get('url', '')}\n\n"
            )
        else:
            contexto_texto += (
                f"Tipo: Registro bibliográfico\n"
                f"Título: {c.get('titulo', '')}\n"
                f"Clasificación: {c.get('clasificacion', '')}\n"
                f"Biblioteca: {c.get('biblioteca', '')}\n"
                f"Ubicación: {c.get('ubicacion', '')}\n"
                f"URL: {c.get('url', '')}\n\n"
            )

    return contexto_texto

def responder(pregunta: str) -> str:
    contexto = buscar_contexto(pregunta, k=5)
    contexto_texto = construir_contexto_texto(contexto)

    prompt = f"""
Eres el asistente virtual de una biblioteca universitaria.

Responde de forma clara, breve y útil, usando únicamente la información del contexto.
Si la respuesta no está en el contexto, dilo claramente.

Contexto:
{contexto_texto}

Pregunta del usuario:
{pregunta}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente útil de biblioteca."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()