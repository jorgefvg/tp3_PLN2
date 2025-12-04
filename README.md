# Chatbot RAG Multi-Agente para Consulta de CVs

Este proyecto implementa un chatbot inteligente capaz de responder preguntas sobre tres currículums diferentes utilizando:

- RAG (Retrieval-Augmented Generation)
- Agentes especializados (uno por CV)
- Pinecone como base vectorial
- Embeddings de HuggingFace
- LangChain para implementar un chunking inteligente
- Groq LLaMA 3.3 70B para generación de texto
- Streamlit para la interfaz del chatbot

El chatbot identifica automáticamente a qué persona corresponde la pregunta, usa el agente adecuado, consulta sus vectores y genera la respuesta.

https://github.com/user-attachments/assets/bc9e78c8-88ba-44a7-b1d6-a0c0b54e4645

```
Estructura del proyecto:
tp3/
│
├── upload_pdf.py     ← Carga los 3 CV, aplica chunking, embeddings de HF y sube los vectores a Pinecone
├── chat_rag.py       ← Chatbot multi-agente
├── environment.yml   ← Creacion del entorno con Conda
├── Jorge_cv.pdf      ← Curriculum propio
├── Francisco_cv.pdf  ← Curriculum de otra persona
├── Ricardo_cv.pdf    ← Curriculum de otra persona
└── README.md           
```

El proyecto se divide en dos scripts principales:

En upload_pdf.py:
1. Se sube en formato pdf los **3 CV**.
2. Se extrae el texto.
3. Se aplica **chunking** usando LangChain.
4. Cada chunk se convierte en un embedding mediante **HuggingFace / SentenceTransformers**.
5. Los vectores se guardan en **Pinecone**, agregando metadata:
    {
    "text": "...chunk...",
    "owner": "Jorge" | "Ricardo" | "Francisco"
    }

En chat_rag.py:
1. Se implementa un sistema de agentes, donde cada agente tiene su propio nombre, identificador para su metadata, un alias para que el chatbot lo detecte en preguntas y un metodo de busqueda en Pinecone.
2. Se implementa un Conditional Edge que se encarga del proceso de analizar la pregunta, por ejemplo:

    - Si detecta “Ricardo”, usa solo ese CV.
    - Si detecta “Ricardo y Francisco”, usa ambos CVs.
    - Si no aparece ningún nombre, usa Jorge por defecto.

3. Se utiliza un RAG Multi-Agente, donde por cada agente involucrado:
    - Se genera embedding de la pregunta
    - Se consulta Pinecone solo dentro de su metadata
    - Se obtienen los chunks relevantes
    - Se concatenan todos los contextos
    - Se envían a Groq para generar una respuesta unificada

---

## Tecnologías utilizadas

- **Chunking:** LangChain – `RecursiveCharacterTextSplitter`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB:** Pinecone Serverless
- **LLM:** Groq – Llama 3.3 70B Versatile
- **Frontend:** Streamlit
- **PDF parsing:** PyPDF2
- **Regex** (re.match)

---

## Cómo utilizar este proyecto

### Prerrequisitos
- Anaconda o miniconda
- Git
- Visual Studio Code

### 1. Configurar variables de entorno:
- Tener una API KEY de base de datos vectoriales. En este proyecto se utilizo el "free tier" de Pinecone (https://www.pinecone.io/).
- Tener una API KEY para utilizar modelos de embeddings y LLMs. En este proyecto se utilizo Groq, que cuenta con una capa gratuita (https://groq.com/)
- Agregar las KEYs a las variables de entorno del sistema operativo para poder llamarlas desde Python.
    
    PINECONE_API_KEY=<TU_CLAVE_PRIVADA>

    GROQ_API_KEY=<TU_CLAVE_PRIVADA>

- Clonar este repositorio o descargarlo.

### 2. Crear y activar el environment:

```bash
conda env create -f environment.yml
conda activate pln2-env
```

### 3. Carga los 3 CV, aplica chunking, embeddings de HF y sube los vectores a Pinecone
```bash
python upload_pdf.py
```

### 4. Ejecutar el chatbot
```bash
streamlit run chat_rag.py
```
Aparecerá una interfaz web en donde se pueden hacer preguntas sobre los 3 cv como:

- “¿Cuál es la experiencia laboral de Francisco?”

- “¿Cuál es tu experiencia profesional?”

- “¿Qué titulos de grado universitario tienen Ricardo y Francisco?” etc ....

nota: si la pregunta NO menciona ningún nombre, entonces se entiende que se esta preguntando por Jorge.
