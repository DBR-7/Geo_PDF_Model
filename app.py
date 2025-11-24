import streamlit as st
import os
import tempfile
from groq import Groq

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =======================================
#              CONFIG
# =======================================
BASE_KNOWLEDGE_PATH = "MINI_Radhakrishnan-sir.pdf"  # Your GIS base document

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# =======================================
#   CACHED BASE GIS LOADER (TOP-LEVEL)
# =======================================
@st.cache_resource
def load_base_gis(embedding_model, pdf_path):
    """
    Loads the base GIS PDF into a FAISS vector DB.
    IMPORTANT: This must NOT be a class method.
    """

    if not os.path.exists(pdf_path):
        st.error(f"Base PDF missing in repo: {pdf_path}")
        return None

    with st.spinner("Loading GIS Reference Document…"):
        docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embedding_model)

    st.success(f"GIS Document Loaded Successfully ({len(chunks)} chunks)")
    return db

# =======================================
#         DUAL RAG ENGINE (GIS)
# =======================================
class DualGeospatialRAG:

    def __init__(self):
        # Embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )

        # Groq client
        if not GROQ_API_KEY:
            st.error("Missing GROQ_API_KEY")
            st.stop()

        self.client = Groq(api_key=GROQ_API_KEY)

        # Session vector DBs
        if "base_db" not in st.session_state:
            st.session_state.base_db = None
        if "user_db" not in st.session_state:
            st.session_state.user_db = None

        # Load GIS document only once (cached)
        if st.session_state.base_db is None:
            st.session_state.base_db = load_base_gis(
                self.embeddings,
                BASE_KNOWLEDGE_PATH
            )

    def process_user_upload(self, uploaded_file):
        if uploaded_file is None:
            st.session_state.user_db = None
            return "No file selected."

        try:
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name

            docs = PyPDFLoader(temp_path).load()
            os.unlink(temp_path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(docs)

            st.session_state.user_db = FAISS.from_documents(
                chunks, self.embeddings
            )

            return f"Loaded {len(chunks)} chunks from {uploaded_file.name}"

        except Exception as e:
            st.session_state.user_db = None
            return f"Error loading PDF: {e}"

    def retrieve(self, query):
        context_parts = []

        # --- BASE GIS DOC ---
        if st.session_state.base_db:
            try:
                base_results = st.session_state.base_db.similarity_search(query, k=3)
                text = "\n".join(doc.page_content for doc in base_results)
                context_parts.append(f"--- BASE GIS DOCUMENT ---\n{text}")
            except:
                pass

        # --- USER DOC ---
        if st.session_state.user_db:
            try:
                usr_results = st.session_state.user_db.similarity_search(query, k=3)
                text = "\n".join(doc.page_content for doc in usr_results)
                context_parts.append(f"--- USER DOCUMENT ---\n{text}")
            except:
                pass

        return "\n\n".join(context_parts)

    def chat(self, message):
        context = self.retrieve(message)

        if context:
            system_msg = (
                "You are a Senior GIS and Geospatial Analytics Expert. "
                "Answer ONLY using the provided context. "
                "For each answer, clearly explain reasoning based on the retrieved context and display relevant excerpts."
            )
            user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {message}"
            st.expander("Context Used").markdown(context)
        else:
            system_msg = "You are a GIS/Geospatial expert. No documents available."
            user_msg = message

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"API Error: {e}"


# =======================================
#           STREAMLIT UI
# =======================================
st.set_page_config(page_title="Geospatial RAG Assistant", layout="centered")
st.title("🗺️ Geospatial RAG Assistant")

# Initialize RAG engine
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = DualGeospatialRAG()

rag = st.session_state.rag_engine

# -------------------------
#     SIDEBAR UPLOAD
# -------------------------
with st.sidebar:
    st.header("Upload GIS/Geospatial PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Process PDF"):
        msg = rag.process_user_upload(uploaded)
        st.info(msg)

# -------------------------
#     CHAT HISTORY
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
#         CHAT BOX
# -------------------------
if prompt := st.chat_input("Ask about GIS, remote sensing, or geospatial..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag.chat(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})





