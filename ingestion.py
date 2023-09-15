import os
import getpass

from langchain.document_loaders import ReadTheDocsLoader,RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'


def ingest_docs()->None:
    loader=RecursiveUrlLoader(url='https://python.langchain.com/docs/modules/')
    raw_documents=loader.load()
    print(f'loaded {len(raw_documents)} documents')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100,separators=["\n\n","\n"," ",""])
    documents=text_splitter.split_documents(documents=raw_documents)
    print(f'Splitted into {len(documents)} chunks')

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(
        documents, embeddings
    )
    print("****** Added to FAISS vectorstore vectors")

    db.save_local("doc_helper_faiss_index")
    

if __name__ =='__main__':
    ingest_docs()