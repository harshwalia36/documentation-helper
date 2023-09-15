import os
from typing import Any

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS



def run_llm(query: str) -> Any:
    embeddings = HuggingFaceEmbeddings()
    docsearch = FAISS.load_local("doc_helper_faiss_index", embeddings=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))