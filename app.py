
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
import os
import time

# Streamlit UI setup
st.set_page_config(page_title="MediChatBot", page_icon="💬", layout="wide")

# Sidebar - Model Configuration
st.sidebar.header("⚙️ Settings")
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.4, 0.1)
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("### 🔧 Additional Settings")
chat_mode = st.sidebar.radio("Chat Mode", ["Concise", "Detailed"])


st.markdown("""
    <h2 style='text-align: center;'>MediChatBot 🤖</h2>
    <p style='text-align: center;'>Your AI-powered medical assistant</p>
    <hr>
""", unsafe_allow_html=True)

# Load Pinecone Index and Embeddings
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medicalchatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load LLM
llm = OllamaLLM(model="llama3.2", temperature=temperature)

# Define Prompt Template
system_prompt = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.

{context}
"""
concise_prompt = system_prompt + "Use three sentences maximum and keep the answer concise."
detailed_prompt = system_prompt + "Provide a detailed explanation with supporting information."

prompt_template = concise_prompt if chat_mode == "Concise" else detailed_prompt

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template),
    ("human", "{input}")
])

# Create RAG Chain
question_and_answer = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_and_answer)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Ask me anything about medical topics...")

if user_input:
    # Reflect user input instantly
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    st.rerun()

# Display chat messages from history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])

if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": st.session_state.chat_history[-1]["message"]})
        bot_response = response["answer"]
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        typing_text = ""
        for char in bot_response:
            typing_text += char
            response_placeholder.markdown(typing_text)
            time.sleep(0.03)
    
    # Save assistant response
    st.session_state.chat_history.append({"role": "assistant", "message": bot_response})
    st.rerun()



