import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

# Loading GROQ and Google API key
groq_api_key= os.getenv("GROQ_API_KEY")

st.title("Gemma Model Document Q&A")

# Loading Model
llm= ChatGroq(groq_api_key= groq_api_key,
              model= "llama3-8b-8192")

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

def vector_embedding():

    if "vectores" not in st.session_state:
        st.session_state.embeddings= GoogleGenerativeAIEmbeddings(model= "models/embeddings-001")
        # Data Ingestion
        st.session_state.loader= PyPDFDirectoryLoader("./data")
        # Loads all documents
        st.session_state.docs= st.session_state.loader.load()
        # Splits loaded documents into chunks
        st.session_state.splitter= RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200)
        st.session_state.documents= st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors= FIASS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


prompt1 = st.text_input("Whst do you want to as from the documents ?")

if st.button("Create Vector Store"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

import time

if prompt1:
    document_chain= create_stuff_documents_chain(llm, prompt)
    retriever= st.session_state.vectors.as_retriever()
    retrieval_chain= create_retrieval_chain(retriever, document_chain)

    start= time.process_time
    response= retrieval_chain.invoke({'input': prompt1})
    st.write(response['answer'])

    with st.expander('Document Similarity Search'):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------------------")
