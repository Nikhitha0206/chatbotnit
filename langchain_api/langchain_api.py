from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI

app = FastAPI()

class Query(BaseModel):
    question: str

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the FAISS vector store from disk
#vectorstore = FAISS.load_local("faiss_index", embeddings)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create retriever and QA chain
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=retriever)

@app.post("/ask")
async def ask_question(query: Query):
    response = qa.run(query.question)
    return {"response": response}
