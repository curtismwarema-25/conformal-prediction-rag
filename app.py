import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

# Load the FAISS index
index = faiss.IndexFlatL2(1536)  # Example: Using OpenAI embedding dimensions
vector_store = FAISS(index, OpenAIEmbeddings())

# Initialize LLM
llm = OpenAI(model_name="gpt-4")

def query_rag_system(question):
    """Searches vector store and generates a response."""
    docs = vector_store.similarity_search(question)
    response = llm(docs[0].page_content) if docs else "No relevant documents found."
    return response

if __name__ == "__main__":
    question = "What is conformal prediction?"
    print(query_rag_system(question))
