import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

def initialize_pinecone(api_key: str, index_name: str, dimension: int = 384, metric: str = "cosine"):
    """
    Initializes and creates a Pinecone index if it doesn't exist.

    :param api_key: Pinecone API key.
    :param index_name: Name of the Pinecone index.
    :param dimension: Embedding dimension size.
    :param metric: Distance metric ("cosine", "euclidean", etc.).
    :return: Pinecone client and index name.
    """
    try:
        if not api_key:
            raise ValueError("Pinecone API key is missing!")

        logging.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=api_key)

        if index_name not in pc.list_indexes():
            logging.info(f"Creating a new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logging.info(f"Index {index_name} created successfully.")
        else:
            logging.info(f"Index {index_name} already exists. Skipping creation.")

        return pc, index_name

    except Exception as e:
        logging.error(f"Error initializing Pinecone: {e}")
        return None, None

def main():
    try:
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables!")

        # Load PDF files and preprocess text
        extracted_data = load_pdf_file(data="data/")
        text_chunks = text_split(extracted_data)
        embeddings = download_hugging_face_embeddings()

        if not embeddings:
            raise RuntimeError("Failed to initialize embeddings model!")

        # Initialize Pinecone and create index
        pc, index_name = initialize_pinecone(PINECONE_API_KEY, index_name="medicalchatbott")

        if not pc or not index_name:
            raise RuntimeError("Failed to initialize Pinecone.")

        # Store documents in Pinecone
        logging.info("Storing documents in Pinecone vector store...")
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings,
        )
        logging.info("Documents successfully stored in Pinecone.")
        return docsearch

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
