import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import (
    extract_text_from_pdf,
    split_documents
)

from src.vector_store import add_documents_to_vectorstore

from src.rag_pipeline import generate_answer


# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="📚",
    layout="wide"
)


# Create folders
os.makedirs("data/uploaded_pdfs", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# Application title
st.title("📚 AI Document Chatbot")

st.write(
    "Upload your PDF documents and ask questions "
    "using Retrieval Augmented Generation (RAG)."
)


# Sidebar
with st.sidebar:

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "Process Documents",
        use_container_width=True
    ):

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")

        else:

            for uploaded_file in uploaded_files:

                file_key = (
                    uploaded_file.name,
                    uploaded_file.size
                )

                if file_key in st.session_state.processed_files:
                    st.info(
                        f"{uploaded_file.name} already processed."
                    )
                    continue

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    unique_name = (
                        f"{uuid.uuid4().hex}.pdf"
                    )

                    pdf_path = os.path.join(
                        "data/uploaded_pdfs",
                        unique_name
                    )

                    # Save uploaded PDF
                    with open(pdf_path, "wb") as file:
                        file.write(uploaded_file.getbuffer())

                    # Extract PDF text
                    documents = extract_text_from_pdf(
                        pdf_path,
                        uploaded_file.name
                    )

                    if not documents:
                        st.warning(
                            f"No pages found in {uploaded_file.name}."
                        )
                        continue

                    # Split documents into chunks
                    chunks = split_documents(documents)

                    if not chunks:
                        st.warning(
                            f"No text extracted from {uploaded_file.name}."
                        )
                        continue

                    # Store embeddings in ChromaDB
                    add_documents_to_vectorstore(chunks)

                    st.session_state.processed_files.add(file_key)

                    st.success(
                        f"{uploaded_file.name} processed successfully!"
                    )


    st.divider()

    st.subheader("⚙️ Controls")

    if st.button(
        "Clear Chat History",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# Main chatbot
st.subheader("💬 Ask Questions About Your Documents")

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(
                        f"📄 {source['file']} — "
                        f"Page {source['page']}"
                    )


# Chat input
question = st.chat_input(
    "Ask something about your uploaded documents..."
)


if question:

    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI answer
    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:

                answer, source_documents = generate_answer(
                    question
                )

                st.markdown(answer)

                # Collect source information
                sources = []

                seen_sources = set()

                for document in source_documents:

                    filename = document.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = document.metadata.get(
                        "page",
                        0
                    ) + 1

                    source_key = (filename, page)

                    if source_key not in seen_sources:

                        seen_sources.add(source_key)

                        sources.append(
                            {
                                "file": filename,
                                "page": page
                            }
                        )

                # Display citations
                if sources:

                    with st.expander("View Sources"):

                        for source in sources:

                            st.write(
                                f"📄 {source['file']} "
                                f"— Page {source['page']}"
                            )

                # Save assistant message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )
                