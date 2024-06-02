from llm_templates import Formatter, Conversation, Content
from prompt_template_utils import system_prompt
messages = [Content(role="system", content=system_prompt),
            Content(role="user", content="Hello!"),
            Content(role="assistant", content="How can I help you?"),
            Content(role="user", content="Write a poem about the sea")]

conversation = Conversation(model='llama3', messages=messages)
conversation_str = Formatter().render(conversation, add_assistant_prompt=True)

print(conversation_str)


# import requests

# def stream_url(url, payload):
#     """Stream data from the given URL and print each part of the response."""
#     headers = {'Content-Type': 'application/json'}
#     # Make a POST request and stream the response
#     with requests.post(url, json=payload, headers=headers, stream=True) as response:
#         try:
#             for line in response.iter_lines():
#                 if line:
#                     decoded_line = line.decode('utf-8')
#                     print(decoded_line)
#         except KeyboardInterrupt:
#             print("Stream was interrupted by the user.")
#         except requests.exceptions.RequestException as e:
#             print(f"Request Error: {e}")
#         except Exception as e:
#             print(f"Error: {e}")

# # URL from your Flask application
# url = "http://localhost:5000/chat_stream"

# # Example payload based on your Flask endpoint
# payload = {
#     "prompt": "What is deterministic encryption?",
#     "history": []
# }

# # Call the function to stream the data
# stream_url(url, payload)
