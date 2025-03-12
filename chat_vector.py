from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from yaml import CBaseLoader

load_dotenv()

# Initialize Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# Create FAISS Vector Store for Chat History
chat_history_vector_db = FAISS(embedding_model)

# Function to store chat history in FAISS
def store_chat_history(user_input, ai_response):
    chat_text = f"User: {user_input}\nAI: {ai_response}"
    chat_history_vector_db.add_documents([Document(page_content=chat_text)])

# Function to retrieve relevant past chat history
def get_relevant_history(user_input):
    docs = chat_history_vector_db.similarity_search(user_input, k=3)  # Fetch top 3 relevant messages
    return "\n".join([doc.page_content for doc in docs])

# Function to load documents into FAISS
def getDocs():
    docLoader = CBaseLoader("https://python.langchain.com/docs/expression_language/")
    docs = docLoader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def retriveChain(docs):
    vectorsrc = FAISS.from_documents(documents=docs, embedding=embedding_model)
    return vectorsrc

def createChain(vectorStore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        ("system", "Relevant chat history: {chat_history}"),  # Inject retrieved chat history
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to retrieve relevant context.")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def process_chat(chain, question):
    relevant_history = get_relevant_history(question)  # Retrieve past chat history

    response = chain.invoke({
        "input": question,
        "chat_history": relevant_history  # Inject relevant history only
    })

    return response["answer"]

if __name__ == '__main__':
    docs = getDocs()
    vectorStore = retriveChain(docs)
    chain = createChain(vectorStore)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(chain, user_input)
        store_chat_history(user_input, response)  # Store conversation in FAISS

        print("Assistant:", response)
