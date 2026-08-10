from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

with open("sample.txt",'r',encoding="utf-8") as file:
    text=file.read()
document=Document(page_content=text)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks=splitter.split_documents([document])
print("="*50)
print("Document Processing")
print("="*50)
print(f"Original characters : {len(text)}")
print(f"Chunks created : {len(chunks)}")

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="semantic_search",
    # persist_directory="chroma_db"----to store on disk
)

print("\nVector store created successfully!")

query=input("\nEnter your search query : ")
result=vector_store.similarity_search(
    query,
    k=3
)
print("\n"+"="*50)
print("Search Results")
print("="*50)

for i,r in enumerate(result,start=1):
    print(f"\nResult {i}")
    print("-"*50)
    print(r.page_content)
