from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks: list):
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"page": chunk["page"]} for chunk in chunks]

    # FastEmbed is a lightweight, faster alternative to sentence-transformers
    # It does not require large torch downloads.
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="db"
    )

    return db