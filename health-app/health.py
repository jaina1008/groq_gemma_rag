# Health Management App
from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import inspect

# Load environment variables from .env
load_dotenv()

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Google Gemini Pro Vision API and get response
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash-002')
    response = model.generate_content([input, image[0], prompt])
    return response.text


# Load input image
def input_image_setup(uploaded_file):
    # Check if a file has ben uploaded
    if uploaded_file:
        # Read file as bytes
        bytes_data = uploaded_file.getvalue()
        # Retrieve mime type of file
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                'data': bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")


# Initialise Streamlit App
st.set_page_config(page_title="Gemini Health App")
st.header("Gemini Health App")
input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an Image: ", type=['jpg',
                                                            'jpeg',
                                                            'png'])
image = ""
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

submit = st.button("Tell me the total calories")

input_prompt = """
You are an expert nutritionist who can see the food items from an image and calculate the total calories. Also provide the details of every food item with calorie intake in the format below:
1. Item 1 - Number of calories
2. Item 2 - Number of calories
---
---

"""

# Show response when submit button is pressed
if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)
    st.subheader("Response: ")
    st.write(response)

# print(inspect.signature(st.set_page_config).parameters.items())
