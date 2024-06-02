import logging
import os
import shutil

import torch
from flask import Flask, jsonify, request, stream_with_context
import asyncio
from constants import *
from utils import *
from pydantic import BaseModel
from typing import List, Any
from flask_ngrok import run_with_ngrok
import threading
from pyngrok import ngrok
import traceback

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
# app.debug = True
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
# Function to run the async loop
def run_async(func, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(func(*args))
    loop.close()
    return result

# Flask route for chat streaming
@app.route("/chat_llm", methods=["POST"])
def chat_llm():
    try:
        # Get the request data
        data = request.get_json()
        question = data.get("prompt")
        history = data.get("history", [])
        print(question, history)
        qa = retrieval_qa_pipline("cuda", llm=LLM, history=history, current_question=question)
        input_data = {
                    "query": question,
                    "chat_history": history
                }

        # Execute the QA pipeline with the prepared inputs
        res = qa(input_data)
        print("====================",res)
        answer, docs = res["result"], res["source_documents"]
        sources = [convert_to_url(docs[0].metadata["source"])]
        return jsonify({"answer": answer, "docs": sources})
    except Exception as e:
        print("Error occured =======",e)
        print(traceback.format_exc())
        return jsonify({"message": f"Error occurred: {str(e)}"}), 500

# Helper function to collect async generator output
async def collect_stream(async_gen):
    chunks = []
    async for chunk in async_gen:
        chunks.append(chunk)
    return chunks

# #########################################################################################################
@app.route("/chat_direct", methods=["POST"])
async def chat_api(chat_request: ChatRequest):
    try:
        data = request.get_json()
        question = data.get("prompt")
        history = data.get("history", [])
        result = await chat(prompt, history)
        return result
    except Exception as e:
        print(e)
        raise e


##########################################################################################################################################s

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    @stream_with_context
    def generate():
        try:
            data = request.get_json()
            question = data.get("prompt")
            history = data.get("history", [])
            print("Received question: ", question)
            yield "Data received, processing...\n"
            
            qa = retrieval_qa_pipline("cuda", True, llm=LLM, promptTemplate_type="llama3")

            res = qa(question)
            answer, docs = res["result"], res["source_documents"]
            sources = [document.metadata["source"] for document in docs]

            # with AsyncCallbackHandler():
            #     for response in qa(question):
            #         yield response['result']
            
            # # Assuming the complete answer is ready to be sent back
            # response = jsonify({"answer": answer, "docs": sources})
            # yield response.get_data(as_text=True)
        except Exception as e:
            yield str(jsonify({"message": f"Error occurred: {str(e)}"}))

    return app.response_class(stream_with_context(generate()))


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
    ngrok.set_auth_token("2gtde6QPcIZPmHSSq0ZhCfW3jli_3ANYfarF8RhuJQbFB7ym1")

    public_url = ngrok.connect(5000, hostname="monarch-modest-cod.ngrok-free.app")
    print(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000/\"")
    app.config["BASE_URL"] = public_url

    threading.Thread(target=app.run).start()
