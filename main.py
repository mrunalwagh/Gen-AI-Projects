from fastapi import FastAPI
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, UnstructuredFileLoader
import PyPDF2
import fitz
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from chromadb.utils import embedding_functions
from langchain_openai import AzureOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
import os
import json
from loguru import logger
import mistune
import markdown

# ---------------------------------------------------------- setup ----------------------------------------------------------------------
# Create a FastAPI application instance
app = FastAPI(title="RAG App")

# Azure openAi key
os.environ["AZURE_OPENAI_API_KEY"] = "***************************"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://exploregenaiworkspace.openai.azure.com"

# Initialize an empty list to store previous chat history
chat_history = []

#------------------------------------------------ Open and read the 'config.json' file----------------------------------------------------
with open('config.json', 'r') as f:  
    app_config = json.load(f)        # Load the contents of the file into app_config as a dictionary

# Define a function to read different file formats 
def read_document(document_path):   
    if document_path.endswith('.docx'):     #for word file
        logger.debug(f'Reading document is word file from path: {document_path}')
        loader = Docx2txtLoader(document_path)
        return loader.load()
    elif document_path.endswith('.txt'):    #for text file
        logger.debug(f'Reading document is txt file from path: {document_path}')
        loader = TextLoader(document_path)
        return loader.load()
    elif document_path.endswith('.pdf'):    #for pdf file
        logger.debug(f'Reading document is a PDF file from path: {document_path}')
        loader = UnstructuredFileLoader(document_path)
        return loader.load()
    elif document_path.endswith('.md'):     #for markdown file
        logger.debug(f'Reading document is a Markdown file from path: {document_path}')
        loader = UnstructuredMarkdownLoader(document_path)
        return loader.load()
    else:
        raise ValueError("Unsupported file format")

# ------------------------------------------------------- Genral Function ------------------------------------------------------------

def chunk_info_fun():   #Define a function to set up text splitting
    return RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)


def set_up_api(document_path):  #Read the document from the provided path
    document = read_document(document_path)
    text_splitter = chunk_info_fun()   # Set up text splitting
    docs = text_splitter.split_documents(document)  # Split the document into chunks
    embedding_function = SentenceTransformerEmbeddings(model_name=app_config.get("llm_tokenizer", 'all-MiniLM-L6-v2'))
    chroma_db = Chroma.from_documents(docs, embedding_function)    # Create Chroma vector store from the documents
    chroma_db.collection = docs   # Create a collection attribute to store chunks
    total_ids = len(docs)  # Get the total chunk count
    return chroma_db, total_ids  # Return Chroma vector store and total chunk count


def summarize(broad_answer):   
    prompt_message = [
        HumanMessage(role="system", content="Use this Following pieces of context to in short single line and meaningful."), 
        HumanMessage(role="user", content=broad_answer)  # User message containing the broad answer
    ]
    llm_model = AzureChatOpenAI( # Create an AzureChatOpenAI instance
        openai_api_version="2023-03-15-preview",  # Set the OpenAI API version
        deployment_name="gpt35exploration"  # Set the deployment name
    )
    llm_response = llm_model.invoke(prompt_message)  # Invoke the LL model with the prompt message
    return llm_response  # Return the summarized response

# ----------------------------------------------------- Endpoint to process questions --------------------------------------------------
@app.post("/request/{question}") 
def process_question(question: str):
    logger.debug(f"Question received: {question}")  # Log the received question
    chroma_db, _ = set_up_api(app_config.get("source_location", "config.json"))  # Set up ChromaDB
    docs = chroma_db.similarity_search(question)  # Search for similar documents
    broad_answer = docs[0].page_content  # Get the broad answer from the first document
    logger.debug(f"Broad answer: {broad_answer}")  # Log the broad answer

    summary = summarize(broad_answer)  # Summarize the broad answer
    logger.debug(f"Sumarize this in short and meaningfull: {summary}")

    # Store chat history
    chat_history.append({"question": question, "answer": summary.content})

    # Return question, answer, and file used with file location
    return {"Question": question, "Answer": summary.content, "FileUsed": app_config.get("source_location", "config.json")}

# ----------------------------------------------------- Endpoint to get chat-history -----------------------------------------------------
@app.get("/chat-history/") 
def get_chat_history():
    logger.debug(f"Last Two Chat History: {chat_history}")  # Log chat history
    return chat_history      # Return the chat history

# ----------------------------------------------------- Endpoint to get file metadata ----------------------------------------------------

@app.get("/file-metadata/")
def get_file_metadata():
    text_splitter = chunk_info_fun()  # Initialize text splitter
    _, total_ids = set_up_api(app_config.get("source_location", "config.json"))
    chunk_size = text_splitter._chunk_size  # Get chunk size
    chunk_overlap = text_splitter._chunk_overlap  # Get chunk overlap

    # Get file path from config
    file_path = app_config.get("source_location", "config.json")
    file_name = os.path.basename(file_path)

    # chunk information
    chunk_info = {
        "Total id Count":total_ids,
        "chunk Information: chunk_size:": chunk_size,"chunk_overlap": chunk_overlap,  
    }

    # metadata information
    metadata_info = {
        "file_name": file_name,
        "file_path": file_path,
    }
    logger.debug(f"chunks & file_metadata_info: Pass")
    return  chunk_info, metadata_info

# ------------------------------------------------------- Endpoint to get file contain --------------------------------------------------
@app.get("/file-contain/")
def file_Contain():
    document_path = app_config.get("source_location", "config.json")    # Get document path from config
    file_contain = read_document(document_path)        # Read the document content
    logger.debug(f"File content: {file_contain}")      # Log the file content
    return file_contain     # Return the file content

# ----------------------------------------------------------------------------------------------------------------------------------------
