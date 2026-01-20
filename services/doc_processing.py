from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
import pymupdf

def initialise_embeddings_model(model_name:str):
    # function wrapper so I can add more detailed functionality later if necessary
    return HuggingFaceEmbeddings(model_name=model_name)

def create_docs_from_bytes(content:bytes, source:str = "unknown", document_id: int = None) -> list[Document]:
    documents = []

    # We use stream here to read a file from memory instead of from a disk location
    with pymupdf.open(stream=content, filetype="pdf") as f:
        for page_num in range(len(f)):
            page = f[page_num]
            text = page.get_text()

            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                    "page": page_num + 1,
                    "total_pages": len(f),
                    "source": source,
                    "document_id": document_id
                }))

    return documents

def split_loaded_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter().from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

def load_and_split_uploaded_document(content:bytes, source:str = "unknown", document_id: int = None) -> List[Document]:
    loaded = create_docs_from_bytes(content, source, document_id)
    return split_loaded_documents(loaded)

