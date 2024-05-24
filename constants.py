import os

# from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader


# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

NEW_INGEST_FOLDER = f"{ROOT_DIRECTORY}/new_ingest"

MODELS_PATH = "./models"

FEEDBACK_TABLE_PATH = f"{ROOT_DIRECTORY}/FeedBack.db"

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """ Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 8096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE


N_GPU_LAYERS = 100  
N_BATCH = 512
DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Uses 1.5 GB of VRAM (High Accuracy & lower VRAM usage)
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_BASENAME = None

# MODEL_ID = "mistralai/Mixtral-7B-Instruct-v0.1"
# MODEL_BASENAME = None