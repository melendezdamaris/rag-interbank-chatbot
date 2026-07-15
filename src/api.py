import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = FastAPI(title="RAG Interbank Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CARGAR ÍNDICE FAISS
# ─────────────────────────────────────────────
print("🔄 Cargando índice FAISS...")

embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="embed-multilingual-v3.0"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

print("✅ Índice cargado")

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024
)

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = """Eres CopilotoIB, el asistente virtual inteligente de Interbank.
Respondes preguntas sobre productos y servicios financieros basándote ÚNICAMENTE 
en la información de los documentos oficiales de Interbank proporcionados.

REGLAS ESTRICTAS:
- Responde SOLO con información de los documentos. No inventes datos.
- Si la información no está en los documentos, di: "No encontré esa información 
  en los documentos disponibles. Te recomiendo contactar a Interbank directamente."
- Nunca inventes tasas, porcentajes o montos exactos que no estén en el contexto.
- Responde siempre en español, con tono profesional pero amigable.
- Sé conciso: máximo 4 párrafos.

CONTEXTO DE DOCUMENTOS OFICIALES:
{context}

PREGUNTA DEL CLIENTE:
{question}

RESPUESTA:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ─────────────────────────────────────────────
# LCEL CHAIN — forma moderna de LangChain
# El operador | conecta los pasos como una pipeline:
# pregunta → retriever → prompt → llm → parser
# ─────────────────────────────────────────────
def format_docs(docs):
    """Une los chunks recuperados en un solo texto de contexto."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Chain RAG lista")


# ─────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class Source(BaseModel):
    file: str
    page: int

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "RAG Interbank Chatbot",
        "embeddings": "Cohere embed-multilingual-v3.0",
        "llm": "Groq Llama 3.1"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        return ChatResponse(answer="Por favor escribe una pregunta.", sources=[])

    # Recuperar docs relevantes para extraer fuentes
    relevant_docs = retriever.invoke(request.question)

    # Generar respuesta con el chain
    answer = rag_chain.invoke(request.question)

    # Extraer fuentes únicas
    sources_seen = set()
    sources = []
    for doc in relevant_docs:
        key = (
            doc.metadata.get("source_file", "Documento"),
            doc.metadata.get("page", 0)
        )
        if key not in sources_seen:
            sources_seen.add(key)
            sources.append(Source(
                file=doc.metadata.get("source_display", "Documento Interbank"),
                page=doc.metadata.get("page", 0) + 1
            ))

    return ChatResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)