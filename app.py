import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Deja las comillas vacías o usa un texto temporal:
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

NOMBRE_PDF = "Manual de Garantía de Productos de BimBam Buy.pdf"

st.set_page_config(
    page_title="BimBam Buy - Asistente de Garantías",
    page_icon="🛍️",
    layout="wide"
)

# ==========================================
# ESTILOS VISUALES (TAMAÑO DE LETRA E IMÁGENES)
# ==========================================
st.markdown("""
    <style>
    /* Aumentar texto general del chat y cuerpo */
    html, body, [class*="css"], .stMarkdown, p, div {
        font-size: 19px !important;
    }
    
    /* Entrada del chat (cuadro donde escribes) */
    .stChatInput textarea {
        font-size: 18px !important;
    }

    /* Tamaño de letra para el desplegable de fuentes */
    .stExpander p, .stExpander div, .stExpander span {
        font-size: 17px !important;
    }

    /* Subtítulos dentro de las fuentes */
    .stExpander strong {
        font-size: 18px !important;
    }

    /* Títulos principales */
    h1 {
        font-size: 2.5rem !important;
    }
    h2, h3 {
        font-size: 1.8rem !important;
    }

    /* Barra lateral */
    [data-testid="stSidebar"] {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# BARRA LATERAL (BRANDING)
# ==========================================
with st.sidebar:
    # Imagen de la marca más grande (160px)
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=160)
    st.title("BimBam Buy")
    st.caption("Asistente Virtual de Consultas Internas")
    st.markdown("---")
    st.markdown(
        "Este agente utiliza RAG para responder dudas sobre las políticas "
        "y garantías de productos basándose únicamente en la documentación oficial."
    )
    st.markdown("---")
    if st.button("🧹 Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# CEREBRO DEL AGENTE (RAG)
# ==========================================
@st.cache_resource
def inicializar_agente():
    loader = PyPDFLoader(NOMBRE_PDF)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "Eres un asistente de IA interno profesional para colaboradores de BimBam Buy. "
        "Responde la pregunta basándote únicamente en el siguiente contexto extraído "
        "de los documentos internos. Si la información no está en el documento, indícalo amablemente.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

agente = inicializar_agente()

# ==========================================
# INTERFAZ DE CHAT CON MEMORIA Y FUENTES
# ==========================================
st.title("🛍️ Agente de Consultas - BimBam Buy")
st.write("Haz tus preguntas sobre garantías, devoluciones y políticas internas.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "¡Hola! Soy el asistente virtual de BimBam Buy. ¿En qué te puedo ayudar hoy respecto al manual de garantías?",
            "fuentes": []
        }
    ]

# Mostrar historial con sus respectivas fuentes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("fuentes"):
            with st.expander("📄 Ver fuentes y páginas consultadas del manual"):
                for i, doc in enumerate(message["fuentes"], 1):
                    pagina = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Fragmento {i} (Página {pagina}):**")
                    st.write(f'"{doc.page_content[:300]}..."')

# Entrada del usuario
if prompt_usuario := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario, "fuentes": []})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Consultando el manual y verificando fuentes..."):
            respuesta = agente.invoke({"input": prompt_usuario})
            texto_respuesta = respuesta["answer"]
            fuentes_encontradas = respuesta.get("context", [])
            
            st.markdown(texto_respuesta)
            
            if fuentes_encontradas:
                with st.expander("📄 Ver fuentes y páginas consultadas del manual"):
                    for i, doc in enumerate(fuentes_encontradas, 1):
                        pagina = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Fragmento {i} (Página {pagina}):**")
                        st.write(f'"{doc.page_content[:300]}..."')
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": texto_respuesta, 
        "fuentes": fuentes_encontradas
    })