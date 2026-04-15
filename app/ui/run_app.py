import os
import sys
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.core.pdf_parser import extract_text_from_pdf
from app.core.chunking import chunk_documents
from app.core.embeddings import create_vector_store
from app.core.qa import ask_question

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()



# UI
st.set_page_config(page_title="RAG PDF Assistant", page_icon="📄")
st.title("🕵🏻‍♂️ Custom RAG Agent ")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])



# CACHE HEAVY PROCESSING
@st.cache_resource
def process_pdf(file_path):
    pages = extract_text_from_pdf(file_path)
    chunks = chunk_documents(pages)
    db = create_vector_store(chunks)
    return db


if uploaded_file:

    DATA_DIR = os.path.join(os.getcwd(), "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    file_path = os.path.join(DATA_DIR, uploaded_file.name)


    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")


    with st.spinner("Processing PDF..."):
        db = process_pdf(file_path)

    st.success("Ready! Ask your question below 👇")



    query = st.chat_input("Ask a question about the document")

    if query:
        with st.spinner("Thinking..."):
            answer, docs = ask_question(db, query)


        st.chat_message("user").write(query)
        st.chat_message("assistant").write(answer)


        with st.expander("📚 View Sources"):
            for i, d in enumerate(docs):
                st.markdown(
                    f"**Source {i+1} (Page {d.metadata.get('page', 'N/A')}):**"
                )
                st.write(d.page_content)
                st.divider()