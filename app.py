import logging
import os
import shutil
import subprocess
import argparse

import torch
from flask import Flask, jsonify, request, stream_with_context, Response
from main import load_model
import asyncio
from constants import *
from utils import *
from pydantic import BaseModel
from typing import List, Any
from flask_ngrok import run_with_ngrok
import threading
import getpass
from pyngrok import ngrok, conf


class ChatRequest(BaseModel):
    prompt: str
    history: List[Any]

if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"

SHOW_SOURCES = True
global DB
global RETRIEVER
global QA

global chatUser
chatUser = {'name': "N/A", 'org' : "N/A", 'email' : "N/A"}

LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)


from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app)
run_with_ngrok(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# 1. Ingest new docs
# 2. Re ingest (clear DB and reingest all docs again)
# 3. Chat with LLM
# 4. Summarize the conversation
# 5. Feedback storing
# 6. Clear chat history. Fresh chat




########################################  1. Ingest new docs ########################################################################

@app.route("/userinfo", methods=['POST'])
def userinfo():
    chatUser = {'name': "N/A", 'org' : "N/A", 'email' : "N/A"}
    userinfo = request.form
    chatUser['name'] = userinfo['user']
    chatUser['org'] = userinfo['orgInfo']
    chatUser['email'] = userinfo['email']
    logging.info("User info saved. \n {chatUser}")
    return jsonify({"message": "User Info Saved"}), 200

    
########################################## 2. Re ingest #####################################################################
@app.route("/ingest_new", methods=["POST"])
def ingest_new():
    global DB
    global RETRIEVER
    global prompt
    global memory
    global QA
    # if not files are passed
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400

    # gets list of all files passed
    files = request.files.getlist('files')
    unsupported_files=[]
    files_to_ingest = False

    try:
        for file in files:
            # verify if files are supported, save to folder if supported
            if file and allowed_file(file.filename):
                # Process each file here
                file_path = os.path.join(NEW_INGEST_FOLDER, file.filename)
                if not os.path.exists(NEW_INGEST_FOLDER):
                    os.makedirs(NEW_INGEST_FOLDER)
                    file.save(file_path)
                    files_to_ingest = True
            # if files are not supported, adding them to a new list and can be notified in response
            else:
                unsupported_files.append(file)

        # Ingesting the new documents directory
        if files_to_ingest:
            DB, RETRIEVER, prompt, memory, QA, status = ingest(NEW_INGEST_FOLDER, DEVICE_TYPE, LLM, SHOW_SOURCES)
        else:
            logging.info("No Supported files to Ingest. Skipping it")

        # Adding new files to original Source_directory
        if status == 'Completed':
            merge_source_docs(NEW_INGEST_FOLDER, SOURCE_DIRECTORY)
            logging.info("New files are added to Source Directory")
        return jsonify({"message": "All Supported files are Ingested, File Directories updated", "unsupported":unsupported_files}), 200

    except Exception as e:
        return jsonify({"message" : f"Error occurred: {str(e)}"}), 500
    



############################################ 3. Re Ingest all Documents ###################################################################
@app.route("/re_ingest_all", methods=["GET"])
def re_ingest_all():
    global DB
    global RETRIEVER
    global prompt
    global memory
    global QA
    try:
        # if there is a folder named DB, deleting it.
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                os.remove("file_ingest.log")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")
        else:
            print("No previous knowledge base exists")    

        DB, RETRIEVER, prompt, memory, QA, status = ingest(SOURCE_DIRECTORY, DEVICE_TYPE, LLM, SHOW_SOURCES)

        if status == "Completed":
            logging.info("All Source Documents are added to Knowledge Base")

        return jsonify({"message": f"Folder '{PERSIST_DIRECTORY}' Vector DB deleted and  Re-ingested all the documents."})
    except Exception as e:
        return jsonify({"message" : f"Error occurred: {str(e)}"}), 500

###################################################### 3. Chat with LLM ##########################################################
# Wrap the async function to run in an event loop
def start_async_loop(async_gen):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(async_gen)
    loop.close()
    return result

@app.route("/chat_llm", methods=["POST"])
def chat_llm():
    try:
        # Get the request data
        data = request.get_json()
        question = data.get("prompt")
        history = data.get("history", [])
        print(question, history)

        # Prepare the async generator
        async_gen = chat_stream(question, history)

        # Run the async generator in a separate thread
        result = threading.Thread(target=start_async_loop, args=(async_gen,))
        result.start()
        result.join()

        # Stream the response
        def generate():
            for chunk in result:
                yield chunk

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        return jsonify({"message": f"Error occurred: {str(e)}"}), 500


    


############################################## 4. Summarize the conversation #################################################################
@app.route("/chat_summary", methods=["POST"])
def chat_summary():

    return jsonify({"message": f"Folder '{PERSIST_DIRECTORY}' Summary to the chat recieved."})
############################################################# 5. Feedback storing ##################################################
@app.route("/chat_feedback", methods=["POST"])
def chat_feedback():
    try:
        feedback_data = request.form
        store_feedback(chatUser, feedback_data)
        return jsonify({"message": f"Folder '{PERSIST_DIRECTORY}' Feedback for the response recieved."})
    except Exception as e:
        return jsonify({"message" : f"Exception while saving feedback {str(e)}"}), 500

########################################################### 6. Clear chat history. Fresh chat ####################################################
@app.route("/chat_clear", methods=["GET"])
def chat_clear():
    return jsonify({"message": f"Folder '{PERSIST_DIRECTORY}' Conversation Cleared."})
###############################################################################################################







if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/get-started/your-authtoken")
    conf.get_default().auth_token = getpass.getpass()
    ngrok.set_auth_token(conf.get_default().auth_token)

    public_url = ngrok.connect(5000).public_url
    print(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000/\"")
    app.config["BASE_URL"] = public_url

    threading.Thread(target=app.run).start()
