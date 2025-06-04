import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API")  # Use correct env var name

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Streamlit UI
st.title("Chat Bot")
user_input = st.chat_input("Ask me anything:")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Create Groq client
    client = Groq(api_key=groq_api_key)

    # Get streaming response from Groq
    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role":m["role"], "content":m["content"]} for m in st.session_state.messages],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    # Stream and display assistant reply
    response = ""
    response_container = st.empty()

    for chunk in completion:
        delta = chunk.choices[0].delta.content or ""
        response += delta
        response_container.markdown(f"**Bot:** {response}")
        time.sleep(0.05)

    # Append assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})


