from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import CommaSeparatedListOutputParser,StrOutputParser,JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
load_dotenv()

#simple ai agent talk with chat history

def createChain(input,chatHistory):
    llm=ChatGoogleGenerativeAI(temperature=0.6,model='gemini-1.5-flash',max_retries=2)

    prompt=ChatPromptTemplate.from_messages(
        [


            ("system","Answer user's prompt {input}"),
            MessagesPlaceholder(variable_name="chatHistory"),
            ("human","{input}")
        ]
    )
    chain=prompt|llm
    response = chain.invoke({"input": user_prompt,"chatHistory":chatHistory})

    return response


if __name__=='__main__':
    chatHistory=[

    ]

    while True:
        user_prompt=input("You: ")
        if(user_prompt=='exit'):
            break
        response=createChain(user_prompt,chatHistory)
        chatHistory.append(HumanMessage(content=user_prompt))
        chatHistory.append(AIMessage(content=response.content))
        print("AI"+response.content)
    