from llm_templates import Formatter, Conversation, Content
from prompt_template_utils import system_prompt
messages = [Content(role="system", content=system_prompt),
            Content(role="user", content="Hello!"),
            Content(role="assistant", content="How can I help you?"),
            Content(role="user", content="Write a poem about the sea")]

conversation = Conversation(model='llama3', messages=messages)
conversation_str = Formatter().render(conversation, add_assistant_prompt=True)

print(conversation_str)