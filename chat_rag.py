import os
import re
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configuracion

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agente-index")

LLM_MODEL = "llama-3.3-70b-versatile"

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
groq_client = Groq(api_key=GROQ_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Agentes y selección (conditional edge)

# Lista de agentes / personas reconocidas
AGENTS = ["Jorge", "Ricardo", "Francisco"]

# Precompilar patrones regex para detectar nombres en la pregunta
# Se utiliza r'\b(nombre)\b' para evitar coincidencias parciales
PATTERNS = {agent: re.compile(rf"\b{re.escape(agent)}\b", re.IGNORECASE) for agent in AGENTS}

def decide_agents_from_query(query):
    """
    Devuelve una lista de agentes a consultar en base a la query.
    - Si se menciona uno o más nombres, devuelve los mencionados (sin duplicados).
    - Si no se menciona nadie, devuelve ['Jorge'] (agente por defecto).
    """
    found = []
    for agent, patt in PATTERNS.items():
        if patt.search(query):
            found.append(agent)
    if not found:
        return ["Jorge"]   # caso por default
    return found

# RAG por agente

def retrieve_context_for_agent(agent_name, q_emb, top_k=5):
    """
    Recupera top_k chunks desde Pinecone filtrando por metadata.owner==agent_name.
    Devuelve la lista de matches (cada match tiene metadata.text,...)
    """
    filt = {"owner": {"$eq": agent_name}}
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True, filter=filt)
    matches = res["matches"] if isinstance(res, dict) else res.matches
    return matches

def build_prompt_from_contexts(contexts_dict, question):
    """
    contexts_dict: {'Jorge': [chunk1, chunk2,...], 'Ricardo': [...], ...}
    Construye prompt que incluye secciones separadas por persona, para que
    el LLM responda acorde a cada agente. Si hay un solo agente, prompt simple.
    """
    if len(contexts_dict) == 1:
        # un solo agente: usar su contexto directamente
        agent = next(iter(contexts_dict))
        context = "\n\n".join(contexts_dict[agent])
        prompt = f"""Eres un asistente que responde preguntas sobre el CV de {agent}.
Usa solamente la información en el contexto. Si la información no está, responde "No se encuentra en el CV".

### CONTEXTO ({agent})
{context}

### PREGUNTA
{question}

### RESPUESTA
"""
        return prompt

    # si hay múltiples agentes, indicar claramente a cuál se refiere cada sección
    parts = []
    for agent, chunks in contexts_dict.items():
        sec = f"--- CONTEXTO ({agent}) ---\n" + ("\n\n".join(chunks) if chunks else "(sin contexto)")
        parts.append(sec)
    all_context = "\n\n".join(parts)
    prompt = f"""Eres un asistente que puede responder consultando varios CVs ({', '.join(contexts_dict.keys())}).
Para cada parte de la pregunta que corresponda a una persona, responde usando únicamente el contexto de esa persona.
Si la pregunta no corresponde a algún CV, o falta la información, di explícitamente "No se encuentra en el CV correspondiente".

### CONTEXTO
{all_context}

### PREGUNTA
{question}

### RESPUESTA
"""
    return prompt

def rag_query(question):
    # 1) decidir agentes
    agents = decide_agents_from_query(question)

    # 2) embed de la pregunta (una sola vez)
    q_emb = embedder.encode(question).tolist()

    # 3) recuperar contexto por agente
    contexts = {}
    for agent in agents:
        matches = retrieve_context_for_agent(agent, q_emb, top_k=5)
        # extraer texto de metadata (si no hay matches queda lista vacía)
        chunks = []
        if matches:
            for m in matches:
                meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)
                if isinstance(meta, dict):
                    txt = meta.get("text", "")
                else:
                    txt = ""
                if txt:
                    chunks.append(txt)
        contexts[agent] = chunks

    # 4) construir prompt
    prompt = build_prompt_from_contexts(contexts, question)

    # 5) llamar LLM Groq
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return answer, agents, contexts

# Streamlit UI
st.title("Chatbot RAG — Consulta de CV (con agentes)")

st.markdown("""
**Modo de uso**:
- Pregunta sobre un CV. Si mencionas un nombre (Jorge, Ricardo, Francisco) el sistema usará a esa persona.
- Si NO mencionas nombre, asume **Jorge** por defecto.
- Si mencionas más de uno, traerá contexto de todos los mencionados y responderá acorde.
""")

question = st.text_input("Pregunta sobre el CV:")

if st.button("Enviar"):
    if question.strip():
        answer, agents_used, contexts = rag_query(question)
        st.write("**Agentes consultados:**", ", ".join(agents_used))
        st.write("### Respuesta:")
        st.write(answer)
