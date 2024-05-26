# import transformers
# import torch

# model_id = "meta-llama/Meta-Llama-3-8B"

# pipeline = transformers.pipeline(
#     "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
# )
# pipeline("Hey how are you doing today?")

# python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('')"

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)
from constants import EMBEDDING_MODEL_NAME
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_embeddings(device_type="cpu"):
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )
embeddings = get_embeddings("cpu")
db3 = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
query = 'Random Number'
db3.get() 
docs = db3.similarity_search(query)
print(docs[0].page_content)
