from flask import Flask, jsonify, request
from datetime import datetime
from flask_cors import CORS
import time
app = Flask(__name__)
CORS(app)

# from utils import generate_model_prompt,convert_history_format

# Sample data
CUSTOMERS = [
    {
        "timestamp": datetime.now().isoformat(),
        "name": "John Doe",
        "company": "Acme Corp",
        "summary": "\n\nI'm just an __AI__, I don't have a specific \"model\" or identity. However, I am a large language model trained by Meta AI, and my primary function is to generate human-like text based on the input prompt."
    },
    {
        "timestamp": datetime.now().isoformat(),
        "name": "Jane Smith",
        "company": "Beta Inc",
        "summary": "Jane has revolutionized the customer interaction protocol."
    },
    {
        "timestamp": datetime.now().isoformat(),
        "name": "Alice Johnson",
        "company": "Gamma LLC",
        "summary": "Alice has successfully managed the merger with Zeta."
    },
    {
        "timestamp": datetime.now().isoformat(),
        "name": "Bob Brown",
        "company": "Delta Co",
        "summary": "Bob has developed a new framework for project management."
    }
]

feedbacks = [
    {
        "timestamp": "2024-05-31T12:00:00Z",
        "name": "Alice Johnson",
        "company": "Tech Innovations Inc.",
        "userQuery": "Can we mask production Database?",
        "botResponse": "Yes, masking production databases is possible and recommended for protecting sensitive data.",
        "like": True,
        "manualFeedback": "The response was helpful and detailed. Thanks!"
    },
    {
        "timestamp": "2024-05-31T12:30:00Z",
        "name": "Bob Smith",
        "company": "DataSecure LLC",
        "userQuery": "Is masking reversible?",
        "botResponse": "Masking is generally reversible if set up with reversible techniques and proper access controls.",
        "like": False,
        "manualFeedback": "I need more specifics on techniques for reversible masking."
    },
    {
        "timestamp": "2024-05-31T13:00:00Z",
        "name": "Carol White",
        "company": "FastData Corp",
        "userQuery": "What is the minimum table space needed to mask data?",
        "botResponse": "The required table space depends on the volume and type of data but generally, an additional 10-20% should be considered for masking processes.",
        "like": True,
        "manualFeedback": "Thanks for the clarification. It helps with our planning."
    },
    {
        "timestamp": "2024-05-31T13:30:00Z",
        "name": "Dave Lee",
        "company": "SecureTech Solutions",
        "userQuery": "Can you share a sample masking report?",
        "botResponse": "Currently, I can't provide documents directly. Please refer to the documentation on our website for sample reports.",
        "like": False,
        "manualFeedback": "Please add functionality to download sample reports directly."
    },
    {
        "timestamp": "2024-05-31T13:00:00Z",
        "name": "Carol White",
        "company": "FastData Corp",
        "userQuery": "What is the minimum table space needed to mask data?",
        "botResponse": "The required table space depends on the volume and type of data but generally, an additional 10-20% should be considered for masking processes.",
        "like": True,
        "manualFeedback": "Thanks for the clarification. It helps with our planning."
    },
    {
        "timestamp": "2024-05-31T13:30:00Z",
        "name": "Dave Lee",
        "company": "SecureTech Solutions",
        "userQuery": "Can you share a sample masking report?",
        "botResponse": "Currently, I can't provide documents directly. Please refer to the documentation on our website for sample reports.",
        "like": False,
        "manualFeedback": "Please add functionality to download sample reports directly."
    },
    {
        "timestamp": "2024-05-31T13:00:00Z",
        "name": "Carol White",
        "company": "FastData Corp",
        "userQuery": "What is the minimum table space needed to mask data?",
        "botResponse": "The required table space depends on the volume and type of data but generally, an additional 10-20% should be considered for masking processes.",
        "like": True,
        "manualFeedback": "Thanks for the clarification. It helps with our planning."
    },
    {
        "timestamp": "2024-05-31T13:30:00Z",
        "name": "Dave Lee",
        "company": "SecureTech Solutions",
        "userQuery": "Can you share a sample masking report?",
        "botResponse": "Currently, I can't provide documents directly. Please refer to the documentation on our website for sample reports.",
        "like": False,
        "manualFeedback": "Please add functionality to download sample reports directly."
    },
    {
        "timestamp": "2024-05-31T13:00:00Z",
        "name": "Carol White",
        "company": "FastData Corp",
        "userQuery": "What is the minimum table space needed to mask data?",
        "botResponse": "The required table space depends on the volume and type of data but generally, an additional 10-20% should be considered for masking processes.",
        "like": True,
        "manualFeedback": "Thanks for the clarification. It helps with our planning."
    },
    {
        "timestamp": "2024-05-31T13:30:00Z",
        "name": "Dave Lee",
        "company": "SecureTech Solutions",
        "userQuery": "Can you share a sample masking report?",
        "botResponse": "Currently, I can't provide documents directly. Please refer to the documentation on our website for sample reports.",
        "like": False,
        "manualFeedback": "Please add functionality to download sample reports directly."
    },
    {
        "timestamp": "2024-05-31T13:00:00Z",
        "name": "Carol White",
        "company": "FastData Corp",
        "userQuery": "What is the minimum table space needed to mask data?",
        "botResponse": "The required table space depends on the volume and type of data but generally, an additional 10-20% should be considered for masking processes.",
        "like": True,
        "manualFeedback": "Thanks for the clarification. It helps with our planning."
    },
    {
        "timestamp": "2024-05-31T13:30:00Z",
        "name": "Dave Lee",
        "company": "SecureTech Solutions",
        "userQuery": "Can you share a sample masking report?",
        "botResponse": "Currently, I can't provide documents directly. Please refer to the documentation on our website for sample reports.",
        "like": False,
        "manualFeedback": "Please add functionality to download sample reports directly."
    },
    {
        "timestamp": "2024-05-31T13:00:00Z",
        "name": "Carol White",
        "company": "FastData Corp",
        "userQuery": "What is the minimum table space needed to mask data?",
        "botResponse": "The required table space depends on the volume and type of data but generally, an additional 10-20% should be considered for masking processes.",
        "like": True,
        "manualFeedback": "Thanks for the clarification. It helps with our planning."
    },
    {
        "timestamp": "2024-05-31T13:30:00Z",
        "name": "Dave Lee",
        "company": "SecureTech Solutions",
        "userQuery": "Can you share a sample masking report?",
        "botResponse": "Currently, I can't provide documents directly. Please refer to the documentation on our website for sample reports.",
        "like": False,
        "manualFeedback": "Please add functionality to download sample reports directly."
    }
]


@app.route('/summaries')
def get_customers():
    return jsonify(CUSTOMERS)

@app.route('/feedbacks')
def get_feedbacks():
    return jsonify(feedbacks)

@app.route('/chat_llm', methods=['GET', 'POST'])
def agent():
    data = request.json
    # print(generate_model_prompt(data['history']))
    time.sleep(5)
    return jsonify({'docs':"https://google.com", 'answer': "\n\nBased on the documentation, we know that Data Masking requires additional space, which is approximately 5 times the size of the largest table being masked. Since your largest table is 10KB, the additional space required would be:\n\n5 x 10KB = 50KB\n\nSo, an additional 50KB of space will be required on your database to mask the data. This includes both TEMP tablespace (roughly 20KB) and the tablespace where masking is running (roughly 30KB). | Header 1 | Header 2 | \n |---|---| \n | Row 1 Cell 1 | \n Row 1 Cell 2 | \n | Row 2 Cell 1 | \n Row 2 Cell 2 |"})


@app.route('/user')
def user():

    return jsonify({'message' : "user data is saved", "status" : 200})


if __name__ == '__main__':
    app.run(debug=True)  # Run the server in debug mode on port 5000
