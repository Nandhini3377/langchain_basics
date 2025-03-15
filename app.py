import streamlit as st
import os
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv
import tempfile
from PyPDF2 import PdfReader
import io

# Load environment variables
load_dotenv()

class ChatMessage(BaseModel):
    role: str
    content: str

def initialize_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def read_file_content(uploaded_file):
    """Read content from either PDF or text file"""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        return text_content
    else:  # text file
        return uploaded_file.getvalue().decode()

def process_document(file_content, chunk_size=500, chunk_overlap=100):
    """Split document into chunks and show processing details"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
   
    docs = text_splitter.create_documents([file_content])
   
    # Display processing information
    st.write(f"📄 Document processed into {len(docs)} chunks")
    st.write(f"📊 Average chunk size: {sum(len(doc.page_content) for doc in docs) / len(docs):.0f} characters")
   
    return docs

def create_vector_store(docs, embedding_model):
    vector_store = FAISS.from_documents(documents=docs, embedding=embedding_model)
    return vector_store

def create_chat_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
   
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer questions based on the following context:\n\nContext: {context}"),
        ("human", "{input}")
    ])
   
    # Create retrieval chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    chain = create_stuff_documents_chain(llm, prompt)
   
    return chain, retriever

def main():
    st.title("📚 Document Chat Demo")
    st.write("Upload a document (PDF or TXT) and chat with it to see how context is passed to the LLM!")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
   
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
   
    if "chain" not in st.session_state:
        st.session_state.chain = None
       
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # File upload
    uploaded_file = st.file_uploader("Upload a document (TXT or PDF file)", type=["txt", "pdf"])

    if uploaded_file:
        with st.spinner("Reading document..."):
            # Read file content
            file_content = read_file_content(uploaded_file)
       
        with st.spinner("Processing document..."):
            # Process document
            docs = process_document(file_content)
           
            # Initialize embedding model
            embedding_model = initialize_embeddings()
           
            # Create vector store
            st.session_state.vector_store = create_vector_store(docs, embedding_model)
           
            # Create chat chain
            st.session_state.chain, st.session_state.retriever = create_chat_chain(st.session_state.vector_store)
           
            st.success("Document processed successfully!")

    # Display chat interface
    st.markdown("---")
    st.subheader("Chat with your document")
   
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        if not st.session_state.vector_store:
            st.error("Please upload a document first!")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
       
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get relevant context
        relevant_docs = st.session_state.retriever.invoke(prompt)
       
        # Display context being passed to LLM
        with st.expander("View context being passed to LLM"):
            for i, doc in enumerate(relevant_docs, 1):
                st.markdown(f"**Chunk {i}:**")
                st.text(doc.page_content)
                st.markdown("---")

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke({
                    "input": prompt,
                    "context": relevant_docs
                })
                st.write(response)
               
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()