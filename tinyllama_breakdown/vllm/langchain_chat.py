from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from tinyllama_breakdown.templates.prompt_format import (
    GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE,
)


def langchain_inference(input_user: str):
    inference_server_url = "http://localhost:8000/v1"

    chat = ChatOpenAI(
        model="Erland/tinyllama-1.1B-chat-v0.3-dummy-AWQ",
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
        max_tokens=500,
        temperature=0,
        model_kwargs={"stop": ["."]},
    )

    messages = [
        HumanMessage(content=input_user),
    ]
    result = chat(messages)
    print(result.content)
