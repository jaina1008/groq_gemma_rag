import os
import streamlit as st
# LangChain= open-source framework deisgned to simplofy developement of applications that integrate with language tools.
# Chatgroq= a class from langchain_groq for interacting with a specific language model.
from langchain_groq import ChatGroq
# RecursiveCharacterTextSplitter= a class for splitting texts into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# create_stuff_documents_chain= a function to create a chain that combines documents
from langchain.chains.combine_documents import create_stuff_documents_chain
# ChatPromptTemplate= Class to create prompt templates
from langchain_core.prompts import ChatPromptTemplate
# create_retrieval_chain= Function toi create a retrieval chain
from langchain.chains import create_retrieval_chain
# FAISS= A library for efficient similarity search and clustering of dense vectors.
from langchain_community.vectorstores import FAISS
# PyPDFDirectoryLoader= Class for loading PDF documents from a directory.
from langchain_community.document_loaders import PyPDFDirectoryLoader
# GoogleGenerativeAIEmbeddings= Class for generating embeddings using Google Generative AI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# load_dotenv= Function to load environment variables from a '.env' file
from dotenv import load_dotenv
load_dotenv()

# Loading GROQ key and setting Google API key environment variable.
groq_api_key= os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

# Set title of streamlit application
st.title("Gemma Model Document Q&A")

# Initialise a language model using GROQ API Key and specifies the model to use.
llm= ChatGroq(groq_api_key= groq_api_key,
              model= "Llama3-8b-8192")

# Creates a prompt template for generating responses based on the context and user input.
prompt= ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
"""
)

# Function to create vector embeddings and store them in the session state.
def vector_embedding():

    if "vectores" not in st.session_state:
        # Initializes the embeddings model.
        st.session_state.embeddings= GoogleGenerativeAIEmbeddings(model= "models/embedding-001")
        # Loads PDF documents from the specific directory.
        st.session_state.loader= PyPDFDirectoryLoader("./data")
        # Stores the loaded documents.
        st.session_state.docs= st.session_state.loader.load()
        # Splits loaded documents into chunks.
        st.session_state.text_splitter= RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200)
        # Stores the split documents.
        st.session_state.final_documents= st.session_state.text_splitter.split_documents(st.session_state.docs)
        # Creates a FAISS vector store from the split documents and embeddings.
        st.session_state.vectors= FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Text input field for the user to enter a question
prompt1 = st.text_input("What do you want to ask from the documents ?")

import time
# Checks if the user has entered a question.
if prompt1:
    # Creates a document chain using the language model and prompt.
    document_chain= create_stuff_documents_chain(llm, prompt)
    # Retrives the vector store as a retriever.
    retriever= st.session_state.vectors.as_retriever()
    # Creates a retrieval chain using the retriever and document chain.
    retrieval_chain= create_retrieval_chain(retriever, document_chain)
    # Records the start time for processing.
    start= time.process_time
    # Invokes the retrieval chain with the user's input and stores the response.
    response= retrieval_chain.invoke({'input': prompt1})
    # Displays the answer to the user's question.
    st.write(response['answer'])
    # Creates an expander section to display document similarity search results.
    with st.expander('Document Similarity Search'):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write(f"------------{i+1}---------------")

# A button to trigger vector embeddings
if st.button("Create Vector Store"):
    vector_embedding()
    st.write("Vector Store DB is Ready")