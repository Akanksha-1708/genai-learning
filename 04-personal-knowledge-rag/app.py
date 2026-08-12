from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

files = [
    "documents/ai_notes.txt",
    "documents/langchain_notes.txt",
    "documents/projects.txt"
]

documents = []

for file_path in files:

    loader = TextLoader(
        file_path,
        encoding="utf-8"
    )

    documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("=" * 70)
print("DOCUMENT PROCESSING")
print("=" * 70)

print(f"Documents loaded : {len(documents)}")
print(f"Chunks created   : {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="personal_knowledge"
)

print("\nVector store created successfully!")