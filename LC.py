from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH="data/"
def pdf_files(data):
    loader=DirectoryLoader(data,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader

    )
    documents=pdf_files(data=DATA_PATH)
    
def chunks(extracted_data):
    textsplit=RecursiveCharacterTextSplitter(chunk_size=500,
                                             chunk_overlap=50

    )
    text_chunks=textsplit.split_documents(extracted_data=documents)

def embeddings():
    model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return model
model=embeddings()
DB_FAISS="vectorstore/db_faiss"
database=FAISS.from_documents(text_chunks,model)
database.save_local(DB_FAISS)
