import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import time
from dotenv import load_dotenv

# Load GROQ API key
load_dotenv()
groq_api = os.getenv("GROQ_API")

st.set_page_config(page_title="Chat Assistant", layout="centered")

# Custom CSS to mimic the design
st.markdown("""
    <style>
        .chat-container {
            max-width: 700px;
            margin: auto;
            background-color: white;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }
        .chat-header {
            background-color: #7F66FF;
            color: white;
            text-align: center;
            padding: 15px;
            font-size: 24px;
            font-weight: bold;
            border-top-left-radius: 20px;
            border-top-right-radius: 20px;
        }
        .message {
            margin: 10px 20px;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 80%;
            display: inline-block;
            font-size: 16px;
            line-height: 1.4;
        }
        .user {
            background-color: #F0F0F0;
            text-align: right;
            margin-left: auto;
        }
        .bot {
            background-color: #E3DCFF;
            text-align: left;
        }
        .timestamp {
            font-size: 10px;
            color: gray;
            margin-top: 2px;
        }
        .input-box {
            padding: 10px;
            border-top: 1px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown('<div class="chat-header">Intelligent Assistant</div>', unsafe_allow_html=True)

# Load and process document
pdf_path = r"C:\Users\MAC012997\PROJECTS_FOLDER\Chatbot\attention is all you need.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
db = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = db.as_retriever()

# LLM
groq = ChatGroq(model="gemma2-9b-it", api_key=groq_api)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])
qa_chain = create_stuff_documents_chain(llm=groq, prompt=prompt)
chain = create_retrieval_chain(retriever, qa_chain)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "bot",
        "content": "Hello! How can i help you"
    }]

# Display chat history
for msg in st.session_state.messages:
    sender_class = "user" if msg["role"] == "user" else "bot"
    st.markdown(
        f"""
        <div class="message {sender_class}">
            <div>{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True
    )

# Chat input
query = st.chat_input("Type text")

if query:
    #now = time.strftime('%B %d %Y, %I:%M:%S %p')

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.markdown(
        f"""
        <div class="message user">
            <div>{query}</div>
        </div>
        """, unsafe_allow_html=True
    )

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            result = chain.invoke({"input": query})
            response = result["answer"]
            #now = time.strftime('%B %d %Y, %I:%M:%S %p')
            st.session_state.messages.append({
                "role": "bot",
                "content": response,
                #"time": now
            })
            st.markdown(
                f"""
                <div class="message bot">
                    <div>{response}</div>
                </div>
                """, unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container
