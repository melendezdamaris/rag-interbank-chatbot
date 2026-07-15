import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

load_dotenv()

# ─────────────────────────────────────────────
# EMBEDDINGS CON COHERE
# Usamos embed-multilingual-v3.0 porque:
# 1. Está optimizado para español
# 2. Entiende sinónimos y semántica profunda
# 3. Es lo que se usa en producción real
# ─────────────────────────────────────────────

def get_embeddings():
    return CohereEmbeddings(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model="embed-multilingual-v3.0"
    )


def cargar_pdfs(docs_folder: str = "docs") -> list:
    docs_path = Path(docs_folder)
    all_documents = []
    pdf_files = list(docs_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No se encontraron PDFs en {docs_folder}/")

    print(f"\n📄 Encontrados {len(pdf_files)} PDFs:")
    for pdf_path in pdf_files:
        print(f"  Cargando: {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_file"] = pdf_path.name
            doc.metadata["source_display"] = pdf_path.stem.replace("_", " ").title()
        all_documents.extend(documents)
        print(f"  ✅ {len(documents)} páginas cargadas")

    print(f"\n📚 Total: {len(all_documents)} páginas cargadas")
    return all_documents


def dividir_en_chunks(documents: list) -> list:
    """
    chunk_size=800: balance entre contexto suficiente
    y precisión de búsqueda. Chunks muy grandes traen
    información irrelevante. Chunks muy pequeños pierden
    contexto. 800 chars es el sweet spot para docs bancarios.
    
    chunk_overlap=150: los chunks se solapan 150 chars
    para no cortar tablas de tasas o listas de requisitos
    a la mitad.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Dividido en {len(chunks)} chunks")
    return chunks


def crear_indice_faiss(chunks: list):
    print("\n🔢 Creando embeddings con Cohere embed-multilingual-v3.0...")
    print("   (optimizado para español — búsqueda semántica real)\n")

    embeddings = get_embeddings()

    # FAISS.from_documents hace todo en una sola llamada batch a Cohere
    # Mucho más eficiente que llamar una vez por chunk
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("✅ Índice FAISS creado")
    return vectorstore


def guardar_indice(vectorstore, path: str = "faiss_index"):
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)
    print(f"💾 Índice guardado en {path}/")


def run_indexing():
    print("=" * 50)
    print("🏦 RAG INTERBANK — PIPELINE DE INDEXING")
    print("   Embeddings: Cohere embed-multilingual-v3.0")
    print("   LLM: Groq Llama 3.1")
    print("=" * 50)

    documents = cargar_pdfs("docs")
    chunks = dividir_en_chunks(documents)
    vectorstore = crear_indice_faiss(chunks)
    guardar_indice(vectorstore)

    print("\n" + "=" * 50)
    print("✅ INDEXING COMPLETO — Chatbot listo para usar")
    print("=" * 50)


if __name__ == "__main__":
    run_indexing()