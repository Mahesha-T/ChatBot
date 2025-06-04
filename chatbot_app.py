import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
import time


st.title("CHATBOT")
api_key = os.getenv("GROQ_API")

client = Groq(api_key=api_key)

if "ai_model" not in st.session_state:
    st.session_state.ai_model = "Gamma2-9b-It"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me"):
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role":m["role"], "content":m["content"]} for m in st.session_state.messages],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
        
    response = ""
    response_container = st.empty()

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        response += delta
        response_container.markdown(f"**Bot:** {response}")
        time.sleep(0.05)

    #response_container.markdown(response)

    st.session_state.messages.append({"role":"assistant","content":response})
