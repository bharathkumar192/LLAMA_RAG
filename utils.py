import os
import shutil
import csv
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import click
import torch
from langchain_community.vectorstores import Chroma
from constants import *
from langchain.embeddings import HuggingFaceEmbeddings
import sqlite3
from datetime import datetime
from transformers import pipeline, TextStreamer, GenerationConfig
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
import asyncio
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from load_models import load_full_model
from langchain.chains import RetrievalQA
from prompt_template_utils import get_prompt_template
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from llm_templates import Formatter, Conversation, Content
from prompt_template_utils import system_prompt
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage



def get_embeddings(device_type="cuda"):
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            file_log(file_path + " loaded.")
            loader = loader_class(file_path)
        else:
            file_log(file_path + " document type is undefined.")
            raise ValueError("Document type is undefined")
        return loader.load()[0]
    except Exception as ex:
        file_log("%s loading error: \n%s" % (file_path, ex))
        return None


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    if not filepaths:
        logging.error("No file paths provided.")
        return None
    # Create a thread pool
    with ThreadPoolExecutor(max_workers=min(len(filepaths), os.cpu_count() or 1)) as executor:
        # Load files
        futures = {executor.submit(load_single_document, filepath): filepath for filepath in filepaths}
        data_list = []
        # Collect data
        for future in as_completed(futures):
            filepath = futures[future]
            try:
                result = future.result()
                if result is None:
                    file_log(filepath + " failed to process.")
                else:
                    data_list.append(result)
            except Exception as exc:
                file_log(f"Error processing file {filepath}: {str(exc)}")
        # Return data and file paths
        return (data_list, filepaths)



def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log("Exception: %s" % (ex))

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            if file_extension == ".py":
                python_docs.append(doc)
            else:
                text_docs.append(doc)
    return text_docs, python_docs



def log_to_csv(question, answer):

    log_dir, log_file = "local_chat_history", "qa_log.csv"
    # Ensure log directory exists, create if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full file path
    log_path = os.path.join(log_dir, log_file)

    # Check if file exists, if not create and write headers
    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer"])

    # Append the log entry
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in DOCUMENT_MAP.keys()




def ingest(path,device_type, LLM, SHOW_SOURCES=True):
    # Load documents and split in chunks
    device_type="cuda"
    path=SOURCE_DIRECTORY
    logging.info(f"Loading documents from {path}") 
    documents = load_documents(path)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {path}")
    logging.info(f"Split into {len(texts)} chunks of text")

    embeddings = get_embeddings(device_type)
    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    DB = Chroma.from_documents(texts,embeddings,persist_directory=PERSIST_DIRECTORY,client_settings=CHROMA_SETTINGS)

    RETRIEVER = DB.as_retriever()
    prompt, memory = get_prompt_template(promptTemplate_type="llama3", history=True)
    
    QA = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=RETRIEVER,
        return_source_documents=SHOW_SOURCES,
        chain_type_kwargs={
            "prompt": prompt,
            "memory": memory
        },
    )
    return DB, RETRIEVER, prompt, memory, QA, "Completed"


def merge_source_docs(source=f"{ROOT_DIRECTORY}/new_ingest", dest=SOURCE_DIRECTORY):
    if not os.path.exists(source):
        print(f"Source directory does not exist: {source}")
        return

    # Ensure the destination directory exists, if not create it
    if not os.path.exists(dest):
        os.makedirs(dest)
        logging.info("Destination path doesnt exist")
        print(f"Destination directory created: {dest}")

    # Walk through all files and directories in the source
    for root, dirs, files in os.walk(source):
        # Move each file
        for file in files:
            src_path = os.path.join(root, file)
            # Create a relative path
            rel_path = os.path.relpath(src_path, source)
            dest_path = os.path.join(dest, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            # Move the file to the destination
            shutil.move(src_path, dest_path)
            print(f"Moved file {src_path} to {dest_path}")


    # Optionally, remove the source directory after moving
    shutil.rmtree(source)
    print(f"Source directory removed: {source}")


def store_feedback(userinfo, feedback_data):
    # Define the database name
    db_name = FEEDBACK_TABLE_PATH

    # Connect to the SQLite database (it will create the file if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Check if the table exists and create it if it does not
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            org TEXT,
            email TEXT,
            user_question TEXT,
            agent_response TEXT,
            like_status BOOLEAN,
            manual_feedback TEXT,
            timestamp DATETIME
        )
    ''')

    # Prepare data for insertion
    chatUser = {
        'name': userinfo.get('user', ''),
        'org': userinfo.get('orgInfo', ''),
        'email': userinfo.get('email', '')
    }
    user_question = feedback_data.get('user', '')
    agent_response = feedback_data.get('agent', '')
    like_status = feedback_data.get('likeStatus', False)
    manual_feedback = feedback_data.get('feedback', '')
    current_timestamp = datetime.now()

    # Insert data into the feedback table
    cursor.execute('''
        INSERT INTO feedback (name, org, email, user_question, agent_response, like_status, manual_feedback, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (chatUser['name'], chatUser['org'], chatUser['email'], user_question, agent_response, like_status, manual_feedback, current_timestamp))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    logging.info(f"Feedback stored successfully for user {chatUser['name']}")


def store_chat_summary(userinfo, chat_summary):
    # Define the database name
    db_name = FEEDBACK_TABLE_PATH

    # Connect to the SQLite database (it will create the file if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Check if the summary table exists and create it if it does not
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            org TEXT,
            email TEXT,
            summary TEXT,
            timestamp DATETIME
        )
    ''')

    # Prepare data for insertion
    username = userinfo.get('user', '')
    org = userinfo.get('orgInfo', '')
    email = userinfo.get('email', '')
    current_timestamp = datetime.now()

    # Insert data into the summary table
    cursor.execute('''
        INSERT INTO summary (username, org, email, summary, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (username, org, email, chat_summary, current_timestamp))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("Chat summary stored successfully.")


def form_history_obj(history):
    new_history = []
    for entry in history:
        if entry["type"] == "ai_response":
            ai_message = AIMessage(entry["message"])
            new_history.append(ai_message)
        elif entry["type"] == "user_prompt":
            human_message = HumanMessage(entry["message"])
            new_history.append(human_message)

    return new_history




async def qa_chain(llm, retriever, chat_history):
    # Convert chat history into the format required by llama3
    formatted_history = form_history_obj(chat_history)
    
    # Set up the prompt template with system and placeholder for user input
    messages = [Content(role="system", content=system_prompt)]
    messages.extend([Content(role="user", content=msg.content) for msg in formatted_history if isinstance(msg, HumanMessage)])
    messages.append(Content(role="assistant", content=""))  

    # Create the conversation object
    conversation = Conversation(model='llama3', messages=messages)
    conversation_str = Formatter().render(conversation, add_assistant_prompt=True)
    
    # Use the existing retrieval and document processing chains but now with the new conversation
    qa_prompt = ChatPromptTemplate.from_template(conversation_str)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


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



async def chat_stream(question: str, chat_history=[]):
    try:
        callback = AsyncIteratorCallbackHandler()
        chain = await qa_chain(callback)
        history = form_history_obj(chat_history)
        input = {"input": question, "chat_history": history}
        task = asyncio.create_task(chain.ainvoke(input=input))
        print("Stream start")
        async for token in callback.aiter():
            print(token)
            yield token

        await task
    except Exception as e:
        print(e)
        raise e