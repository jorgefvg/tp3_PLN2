import os
import uuid
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Configuracion (las API keys se agregan al PATH de windows)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agente-index")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

# Embeddings (HuggingFace)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Configuracion Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,     # MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print("Índice creado:", PINECONE_INDEX_NAME)

index = pc.Index(PINECONE_INDEX_NAME)

# Funciones

def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


def create_chunks(text):
    """
    Chunking inteligente usando LangChain.
    Obtiene bloques coherentes sin mezclar secciones.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=[
            "\n## ", "\n### ", "\n- ", "\n•",
            "\n", ". ", " "
        ]
    )
    return splitter.split_text(text)


def upload_pdf(pdf_path, owner=None):
    """
    Extrae texto, crea chunks, genera embeddings y sube a Pinecone.
    Se agrego metadata 'owner' para poder filtrar por agente (persona).
    """
    if owner is None:
        # inferir owner por nombre de archivo (Jorge_cv.pdf -> Jorge)
        owner = os.path.basename(pdf_path).split("_")[0].capitalize()

    print(f"Procesando '{pdf_path}' (owner={owner})...")
    text = extract_text_from_pdf(pdf_path)

    print("Creando chunks inteligentes...")
    chunks = create_chunks(text)
    print(f"Generados {len(chunks)} chunks correctamente")

    print("Generando embeddings y preparando vectores...")
    vectors = []

    for i, ch in enumerate(chunks):
        emb = embedder.encode(ch).tolist()
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": emb,
            "metadata": {
                "text": ch,
                "owner": owner,
                "source": os.path.basename(pdf_path),
                "chunk_id": i
            }
        })

    print(f"Subiendo {len(vectors)} vectores a Pinecone (index={PINECONE_INDEX_NAME})...")
    # Upsert en batches
    index.upsert(vectors=vectors)
    print("Upload completado. Chunks y embeddings almacenados en Pinecone.")


if __name__ == "__main__":
    # Por defecto, al ejecutar el script sube los 3 CVs:
    # Jorge_cv.pdf, Ricardo_cv.pdf, Francisco_cv.pdf
    files_and_owners = [
        ("Jorge_cv.pdf", "Jorge"),
        ("Ricardo_cv.pdf", "Ricardo"),
        ("Francisco_cv.pdf", "Francisco"),
    ]

    for fpath, owner in files_and_owners:
        if os.path.exists(fpath):
            upload_pdf(fpath, owner=owner)
        else:
            print(f"Advertencia: no existe el archivo {fpath} - omitiendo.")
