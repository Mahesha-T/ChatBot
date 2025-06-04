import streamlit as st
import os
import time
from dotenv import load_dotenv
#from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import re
# Load environment
load_dotenv()
groq_api = os.getenv("GROQ_API")

#st.title("WEB CHATBOT")
st.set_page_config(page_title="PDF QA", layout="centered")
st.title("📄 Chat with MPS Labs")

# Load document
loader = WebBaseLoader("https://www.highwirepress.com/services/mps-labs/")
#loader = WikipediaLoader(query="Mahesh Babu")
pages = loader.load()
#pagess = loader.invoke("Mahesh Babu")
#st.write(pages[0].page_content)
pages = [Document(page_content=text.page_content,metadata=text.metadata) for text in pages]
# Embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
groq = ChatGroq(model="gemma2-9b-it", api_key=groq_api)

# for doc in pages:
#     doc.metadata = filter_complex_metadata(doc.metadata)

# Split and embed
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
docs = splitter.split_documents(pages)
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()

system_prompts = (
    "Use the given context to answer the question. "
    "Always use history to give answers more accurately. "
    "Avoid starting the response with phrases like 'According to the text', 'The text states that', or similar. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)



prompts = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompts),
        ("user","{input}")
    ]
)

llm_prompt = create_stuff_documents_chain(llm=groq,prompt=prompts)

chain = create_retrieval_chain(retriever,llm_prompt)

# QA chain
#qa_chain = RetrievalQA.from_chain_type(llm=groq, retriever=retriever, return_source_documents=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    is_user = role == "user"

    col1, col2 = st.columns([6, 2]) if is_user else st.columns([6, 6])
    with col2 if is_user else col1:
        bg = "#DCF8C6" if is_user else "#F1F0F0"
        speaker = "You" if is_user else "Assistant"
        st.markdown(
            f"""
           <div style="background-color:{bg}; padding:10px 15px; border-radius:12px; margin:8px 0;">
                <b>{speaker}:</b><br>{content}
           
            """,
            unsafe_allow_html=True
        )

# Input
query = st.chat_input("Ask something...")

if query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Show user question immediately
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_msg = st.session_state.messages[-1]["content"]

    with st.spinner("Assistant is typing..."):
        result = chain.invoke({"input": user_msg})
        response = result["answer"]
        response = re.sub(r"According to the text,|The text says that|Based on the text provided,","",response)
        time.sleep(1.0)  # Simulate typing

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
