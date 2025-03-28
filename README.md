# 🏥 MediChatBot - AI-Powered Medical Assistant

MediChatBot is an AI-driven chatbot that extracts, processes, and retrieves medical information using **Pinecone vector search** and **Hugging Face embeddings**. It enables fast, accurate responses to medical queries with a **Streamlit UI**.

## 🚀 Features
- **PDF Processing:** Extracts medical text from PDF documents
- **Text Embeddings:** Uses **Hugging Face sentence-transformers** for efficient vector representation
- **Vector Search:** Stores and retrieves documents via **Pinecone**
- **Streamlit UI:** User-friendly chatbot interface

## 🛠️ Tech Stack
- **LangChain**, **Hugging Face**, **Pinecone**, **Streamlit**, **Python**

## ⚙️ Setup & Installation
### 1️⃣ Clone the Repository & Install Dependencies
```bash
git clone https://github.com/your-username/medical-chatbot.git && cd medical-chatbot
pip install -r requirements.txt
```
### 2️⃣ Set Up API Keys
Create a `.env` file and add your Pinecone API key:
```env
PINECONE_API_KEY=your_pinecone_api_key
```
### 3️⃣ Run the Application
```bash
streamlit run app.py
```

## 🔄 Workflow
📂 **Load PDFs** → 🔍 **Split Text** → 🧠 **Generate Embeddings** → 📡 **Store in Pinecone** → ⚡ **Retrieve & Respond**

## 📜 License & Author
**License:** MIT | **Author:** [R.Sarath Kumar](https://github.com/sarathkumar1304)

