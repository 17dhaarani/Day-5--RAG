import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

# ------------- Hardcoded API Key -------------
GOOGLE_API_KEY = "AIzaSyAnkUkXe4R0kD5ri6lzBB0ln_WTNA08mCY"  # Replace with your real key

# ------------- Streamlit UI ------------------
st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("📄💬 Gemini RAG PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    # Save file to disk for PyPDFLoader
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # ------------- Load + Chunk PDF ------------------
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # ------------- Embed + Store in FAISS ------------------
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # ------------- Define Gemini LLM ------------------
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    # ------------- Prompt Template ------------------
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions based on the given context.
        
        Context:
        {context}

        Question:
        {question}
        """
    )

    # ------------- Build RAG Chain ------------------
    rag_chain = (
        RunnableMap({
            "context": lambda x: retriever.get_relevant_documents(x["question"]),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    # ------------- User Input ------------------
    user_input = st.text_input("Ask a question based on the uploaded PDF:")

    if user_input:
        with st.spinner("🤖 Thinking..."):
            answer = rag_chain.invoke({"question": user_input})
            st.markdown("### 📘 Answer:")
            st.write(answer.content)
