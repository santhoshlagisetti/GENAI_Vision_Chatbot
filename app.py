from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")
 
def get_gemini_response(question):
    res = model.generate_content(question)
    return res.text

st.set_page_config(page_title="Gemini 1.5 Flash Chatbot", page_icon=":robot_face:")
st.title("Gemini 1.5 Flash Chatbot")
st.write("Ask me anything about Gemini 1.5 Flash!")
question = st.text_input("Enter your question here:")
button = st.button("Submit")
if button:
    if question:
        with st.spinner("Generating response..."):
            response = get_gemini_response(question)
            st.write(response)
    else:
        st.warning("Please enter a question.")
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

   