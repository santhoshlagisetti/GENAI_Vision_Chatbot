import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import Google's embeddings and chat models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


st.header("My First Chatbot with Gemini")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

text = ""
if file:
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    # st.write(text) # Uncomment to see the extracted text

if text:
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks) # Uncomment to see the text chunks

    # Initialize Google Generative AI Embeddings
    # Using "models/embedding-001" which is a general-purpose embedding model
    if GOOGLE_API_KEY:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vectorstore = FAISS.from_texts(chunks, embeddings)
    else:
        st.error("Google API Key not found. Please set it as an environment variable or in Streamlit secrets.")
        st.stop() # Stop execution if API key is missing


question = st.text_input("Ask a question about the document?")

if question:
    if 'vectorstore' in locals(): # Ensure vectorstore is created before searching
        docs = vectorstore.similarity_search(question)

        # Initialize ChatGoogleGenerativeAI for the chat model
        # Using "gemini-pro" for general chat
        if GOOGLE_API_KEY:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0, # Set temperature to 0 for more deterministic answers
                # max_tokens=1000, # max_tokens is not a direct parameter for ChatGoogleGenerativeAI
                verbose=True,
            )
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=question)
            st.write(response)
        else:
            st.error("Google API Key not found. Cannot run the LLM chain.")
    else:
        st.warning("Please upload a PDF file first to create the vector store.")

