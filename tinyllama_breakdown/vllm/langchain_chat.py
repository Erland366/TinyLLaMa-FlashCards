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


def langchain_inference():
    example_prompt = """Sekolah menengah atas negeri dan swasta tersebar di berbagai wilayah di tanah air. SMA merupakan salah satu pilihan yang banyak dipilih pelajar selepas lulus SMP. Melanjutkan sekolah SMA lebih berpeluang untuk masuk ke perguruan tinggi negeri dengan memilih jurusan kuliah yang diminatinya.

    Pemilihan jurusan bukan hanya dilakukan pada jenjang pendidikan pada SMK. Namun juga berlaku pada siswa siswi yang memilih masuk ke SMA. Jurusan yang terdapat pada SMA, antara lain: jurusan IPA, IPS dan jurusan bahasa.
    """  # noqa: E50

    inference_server_url = "http://localhost:8000/v1"

    chat = ChatOpenAI(
        model="Erland/tinyllama-1.1B-chat-v0.3-dummy-AWQ",
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
        max_tokens=500,
        temperature=0,
        # model_kwargs={"stop": ["."]},
    )

    messages = [
        HumanMessage(
            content=GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE.format(
                input_user=example_prompt, response=""
            )
        ),
    ]
    result = chat(messages)
    return result
