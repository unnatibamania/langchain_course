import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

# Load all PDFs from the pds/ folder using LangChain's PyPDFLoader
def load_pdfs(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

# Main RAG workflow
def main():
    pdf_folder = os.path.join(os.path.dirname(__file__), "pdfs")
    documents = load_pdfs(pdf_folder)

    if not documents:
        print("No text extracted from PDFs. Exiting.")
        return

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    if not chunks:
        print("No chunks created from documents. Exiting.")
        return

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)

    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, openai_api_key=openai_key),
        chain_type="stuff",
        retriever=retriever
    )

    print("RAG system initialized. Ask your questions. Type 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            print("Exiting...")
            break
        if not query.strip():
            print("Please enter a question.")
            continue
        
        try:
            result = qa.run(query)
            print("Answer:", result)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
