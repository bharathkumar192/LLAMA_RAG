import os
import logging
import click
import torch
import utils
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template
from langchain_community.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
    TextStreamer
)
from utils import *
from load_models import load_full_model
from huggingface_hub import  HfFolder
HfFolder.save_token('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')
from constants import *
from constants import EMBEDDING_MODEL_NAME
from langchain_community.embeddings import HuggingFaceEmbeddings
llm = load_model("cuda", model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

def get_embeddings(device_type="cuda"):
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")
    model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)
    generation_config = GenerationConfig.from_pretrained(model_id)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        repetition_penalty=1.15,
        generation_config=generation_config,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")
    return local_llm


def retrieval_qa_pipline(device_type, use_history,llm ,promptTemplate_type="llama"):
    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)
    # load the llm pipeline

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,  
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,  
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )
    return qa


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
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama3",
    type=click.Choice(
        ["llama3", "llama", "mistral", "non_llama"],
    ),
    help="model type, llama3, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
def main(device_type, show_sources, use_history, model_type, save_qa):

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    qa = retrieval_qa_pipline(device_type, use_history,llm ,promptTemplate_type=model_type)
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        logging.info("Passing question into QA Retriever")

        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources: 
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

        if save_qa:
            utils.log_to_csv(query, answer)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
