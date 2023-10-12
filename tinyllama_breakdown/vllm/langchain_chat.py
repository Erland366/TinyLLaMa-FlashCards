from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

inference_server_url = "http://localhost:5000/v1"

chat = ChatOpenAI(
    model="Erland/tinyllama-1.1B-chat-v0.3-dummy-AWQ",
    openai_api_key="EMPTY",
    openai_api_base=inference_server_url,
    max_tokens=1024,
    temperature=0,
    model_kwargs={"stop": ["."]},
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to Italian."
    ),
    HumanMessage(
        content="Translate the following sentence from English to Italian: I love programming."
    ),
]
result = chat(messages)
print(result)
