import os

from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

from langchain_groq import ChatGroq


from src.vector_store import search_documents


load_dotenv()


def get_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=api_key
    )

    return llm


def generate_answer(question):

    # Step 1: Retrieve relevant documents
    documents = search_documents(question, k=4)

    if not documents:
        return "No relevant information was found.", []

    # Step 2: Prepare context
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page: {doc.metadata.get('page', 0) + 1}\n"
            f"{doc.page_content}"
            for doc in documents
        ]
    )

    # Step 3: Create prompt
    prompt = f"""
    You are an AI document assistant.

    Answer the user's question using ONLY the
    information in the provided context.

    If the answer is not available in the context,
    clearly say that the information was not found
    in the uploaded documents.

    Do not invent facts.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    # Step 4: Generate answer
    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content, documents
