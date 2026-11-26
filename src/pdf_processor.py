from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path, original_filename):
    """Extract text and metadata from a PDF."""

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    for document in documents:
        document.metadata["source"] = original_filename

    return documents


def split_documents(documents):
    """Split documents into smaller text chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    return chunks
