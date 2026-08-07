from dotenv import load_dotenv
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pdf_loader import (
    load_pdf,
    split_documents
)

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=1500,
)

model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are an expert document summarizer.
Analyze the following document and generate:
1. Executive Summary
2. Key Points
3. Important Concepts
4. Keywords
5. Final Takeaways
Document:
{text}
"""
)

parser = StrOutputParser()
chain = prompt | model | parser

def main():
    pdf_path = "sample_pdfs/sample.pdf"
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    print("=" * 70)
    print("PDF LOADED SUCCESSFULLY")
    print("=" * 70)
    print(f"Pages Loaded  : {len(documents)}")
    print(f"Chunks Created: {len(chunks)}")

    full_text = "\n\n".join(
        chunk.page_content
        for chunk in chunks
    )
    print("\nGenerating Summary...\n")
    summary = chain.invoke(
        {
            "text": full_text
        }
    )
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary)

    with open(
        "outputs/summary.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(summary)

    with open(
        "outputs/summary.md",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("# PDF Summary\n\n")
        file.write(summary)
    print("\nSummary saved successfully!")

if __name__ == "__main__":
    main()