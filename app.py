import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Loading GROQ and Google API key
groq_api_key= os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

# Loading Model
llm= ChatGroq(groq_api_key= groq_api_key,
              model= "llama-3.1-8b-instant")
            #   model= "Llama3-8b-8192")

prompt= ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. Whenever possible, show basics name field in each response as a header.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Data Ingestion: Use DirectoryLoader to load JSON files
        st.session_state.loader = DirectoryLoader(
            "./data",
            glob="**/*.json",
            show_progress=True,
            loader_cls=JSONLoader,
            loader_kwargs = {'jq_schema': '(.basics | tostring) + " " + (.work[] | tostring) + " " + (.interests[] | tostring)'}

        )

        # Load all documents
        st.session_state.docs = st.session_state.loader.load()

        # Split loaded documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Generate vectors from the documents
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)




prompt1 = st.text_input("What do you want to ask from the documents ?")

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

    # with st.expander('Document Similarity Search'):
    #     for i, doc in enumerate(response['context']):
    #         st.write(doc.page_content)
    #         st.write("---------------------------")
