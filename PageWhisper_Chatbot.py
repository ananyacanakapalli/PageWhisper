import os
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Page Customization
st.set_page_config(
    page_title="PageWhisper",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #2B7A78;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-heading {
        font-size: 22px;
        margin-top: 25px;
        color: #17252A;
        font-weight: 600;
    }
    .result-box {
        background: #DEF2F1;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


## Sidebar
# API Key
OPENAI_API_KEY = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Please provide your OpenAI API Key either in the sidebar or as an environment variable.")
    st.stop()

# Uploading File
st.sidebar.subheader("Upload your document")
file = st.sidebar.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])


## Main page
# Title
st.markdown("<div class='main-title'>PageWhisper: Ask Your Docs Anything</div>", unsafe_allow_html=True)

# Ask Question
st.subheader("Ask a Question")
question = st.text_input("Type your question:")

# File Processing
if file:
    text = ""

    # PDFs
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    # Docs
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = " ".join([para.text for para in doc.paragraphs])

    # Unsupported types
    else:
        st.error("Unsupported file type. Upload PDF or DOCX.")
        st.stop()

    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Processing User Question
    if question:
        docs = vector_store.similarity_search(question)
        context = "\n\n".join([d.page_content for d in docs])

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Context:
{context}

Question:
{question}

Provide the most accurate answer based on the context above.
"""
        )

        final_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(final_prompt)

        st.markdown("<div class='sub-heading'>AI Answer</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'>{response.content}</div>", unsafe_allow_html=True)

else:
    st.info("Upload a document to begin.")
