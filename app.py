from dotenv import find_dotenv,load_dotenv
from transformers import pipeline
from langchain import PromptTemplate,LLMChain,OpenAI,HuggingFaceHub
import requests
import os
from getpass import getpass
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

def img2text(url):
    image_to_text=pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
    text=image_to_text(url)[0]["generated_text"]
    print(text)
    return text

def generate_story(scenario):
    llm=HuggingFaceHub(repo_id="pspatel2/storygen",
                       model_kwargs={"temperature":0.3,
                                     "max_length":64})
    template="""scenario:{scenario}
    story:"""
    prompt=PromptTemplate(template=template,input_variables=["scenario"])
    story_llm=LLMChain(prompt=prompt,llm=llm)
    story=story_llm.predict(scenario=scenario)
    print(story)
    return story

def text2speech(messages):
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    payloads={
    "inputs":messages
}
    response=requests.post(API_URL,headers=headers,json=payloads)
    with open('audio.flac','wb') as file:
        file.write(response.content)
        
def main():
    st.set_page_config(page_title="img to audio story",page_icon="🔊")
    st.header("turn img into audio story")
    uploaded_file=st.file_uploader("choose a image...",type="jpg")
    if uploaded_file is not None:
        bytes_data=uploaded_file.getvalue()
        with open(uploaded_file.name,"wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file,caption='uploaded image.',
                 use_column_width=True)
        scenario=img2text(uploaded_file.name)
        story=generate_story(scenario)
        text2speech(story)
        
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio("audio.flac")
if "__name__"=="__main__":
    main()