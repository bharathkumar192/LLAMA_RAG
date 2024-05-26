import logging
import click
import torch

import nltk
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

nltk.download('punkt')
from huggingface_hub import  HfFolder
HfFolder.save_token('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')
from constants import *
from utils import *
from constants import EMBEDDING_MODEL_NAME
from langchain_community.embeddings import HuggingFaceEmbeddings



@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "mps",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)

def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    embeddings = get_embeddings(device_type)
    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
