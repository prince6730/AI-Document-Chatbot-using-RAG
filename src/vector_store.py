from langchain_chroma import Chroma

from src.embeddings import get_embedding_model


VECTORSTORE_PATH = "vectorstore"


def get_vectorstore():

    embeddings = get_embedding_model()

    vectorstore = Chroma(
        collection_name="document_chatbot",
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    return vectorstore


def add_documents_to_vectorstore(chunks):

    vectorstore = get_vectorstore()

    vectorstore.add_documents(chunks)

    return vectorstore


def search_documents(question, k=4):

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    return results
