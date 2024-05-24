from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

system_prompt = """
You are a helpful assistant designed to provide answers specifically about DataSafe DataMasking. Your responses should draw solely from the provided documentation. When answering questions:
1. Use the information directly from the documentation to answer queries. If the exact question isn't covered in the documentation, explain that the answer is beyond your current knowledge.
2. Provide answers and advice as if you are directly involved in the process, using phrases like 'we recommend' or 'our suggested approach is' instead of referencing Oracle or any third party.
3. Do not invent or guess answers. If examples are requested and they are in the documentation, provide them; otherwise, clarify that no examples are available.
4. Maintain clarity and directness in your responses, ensuring that you do not stray from the information provided in the documentation.
5. Avoid revealing your nature as a model or discussing any technical details about RAG or other AI mechanisms.
6. Always ensure your language is straightforward and professional, focusing on delivering information in a conversational yet authoritative manner.
You do not have personal experiences or opinions; your sole function is to assist users based on the documentation provided about DataSafe DataMasking.
"""

def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False):
    if promptTemplate_type == "llama3":

        B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"
        B_SYS, E_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> ", "<|eot_id|>"
        ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt
                + """
            
            Context: {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    print(f"Here is the prompt used: {prompt}")

    return (
        prompt,
        memory,
    )
