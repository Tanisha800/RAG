from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(pages: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for page in pages:
        texts = splitter.split_text(page["text"])

        for t in texts:
            chunks.append({
                "content": t,
                "page": page["page"]
            })

    return chunks