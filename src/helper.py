import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_pdf_file(data_path: str):
    """
    Loads all PDF files from the specified directory using PyPDFLoader.

    :param data_path: Path to the directory containing PDF files.
    :return: List of loaded documents.
    """
    try:
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            logging.warning(f"No PDF files found in the directory: {data_path}")
        else:
            logging.info(f"Successfully loaded {len(documents)} document(s) from {data_path}")
        return documents
    except Exception as e:
        logging.error(f"Error loading PDF files from {data_path}: {e}")
        return []

def text_split(extracted_data, chunk_size=500, chunk_overlap=50):
    """
    Splits extracted text into smaller chunks using RecursiveCharacterTextSplitter.

    :param extracted_data: List of extracted documents.
    :param chunk_size: Size of each text chunk.
    :param chunk_overlap: Overlap between consecutive chunks.
    :return: List of text chunks.
    """
    try:
        if not extracted_data:
            logging.warning("No extracted data to split.")
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_chunks = text_splitter.split_documents(extracted_data)

        logging.info(f"Successfully split text into {len(text_chunks)} chunks.")
        return text_chunks
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        return []

def download_hugging_face_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Downloads and initializes Hugging Face embeddings.

    :param model_name: Name of the Hugging Face embedding model.
    :return: HuggingFaceEmbeddings instance.
    """
    try:
        logging.info(f"Downloading Hugging Face embeddings model: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logging.info("Successfully loaded embeddings model.")
        return embeddings
    except Exception as e:
        logging.error(f"Error downloading Hugging Face embeddings: {e}")
        return None
