from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEmbeddings
)
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=1000
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant.
Answer the user's question using ONLY the provided context.
If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents."
Do not invent information.
Context:
{context}
Question:
{question}
Answer:
"""
)

parser = StrOutputParser()
def format_documents(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )

def answer_question(question):
    relevant_documents = retriever.invoke(question)
    context = format_documents(relevant_documents)
    chain = prompt | model | parser
    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )
    return answer

def main():
    print("\n" + "=" * 70)
    print("PERSONAL KNOWLEDGE RAG ASSISTANT")
    print("=" * 70)
    while True:
        question = input(
            "\nAsk a question (type 'exit' to quit): "
        ).strip()
        if question.lower() == "exit":
            print("\nGoodbye!")
            break
        if not question:
            print("Please enter a question.")
            continue
        print("\nSearching knowledge base...")
        answer = answer_question(question)
        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)

if __name__ == "__main__":
    main()