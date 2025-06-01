import os
import google.generativeai as genai
import streamlit as st
from PIL import Image
from dotenv import load_dotenv 
load_dotenv()



genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")
 
def get_gemini_response(input, image=None):
    if input and image:
        get_gemini_response = model.generate_content([input, image])
    elif input:
        get_gemini_response = model.generate_content(input)
    else:
        get_gemini_response = model.generate_content(image)
    return get_gemini_response.text


st.set_page_config(page_title="Gemini 1.5 Vision Chatbot", page_icon=":robot_face:")
st.title("Gemini 1.5 Vision Chatbot")
st.write("Ask me anything about Gemini 1.5 Vision!")
input = st.text_input("Enter your question here:")  
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

button = st.button("Tell me about this image")
if button:
    
    if input and image:
        with st.spinner("Generating response..."):
            response = get_gemini_response(input, image)
            st.write(response)
    elif input :
        with st.spinner("Generating response..."):
            response = get_gemini_response(input)
            st.write(response)
    else:
        st.warning("Please enter a question or upload an image.")   
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
