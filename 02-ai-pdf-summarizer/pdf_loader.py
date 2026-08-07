from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(pdf_path):
    """Load PDF using LangChain."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def split_documents(documents):
    """Split PDF into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)