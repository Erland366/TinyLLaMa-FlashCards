import json
import os

import requests

from tinyllama_breakdown.quantization.awq_quantization import QuantizeAWQ
from tinyllama_breakdown.templates.prompt_format import (
    GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE,
)
from tinyllama_breakdown.utils import TokenCounter, parse_output, read_yaml_file

EXAMPLE_INPUT = """### Instruction:
Saya ingin Anda berperan sebagai pembuat Anki Cards profesional, mampu membuat Anki Cards dari teks yang saya berikan.

Mengenai perumusan isi kartu, Anda berpegang pada dua prinsip:
Pertama, prinsip informasi minimum: Materi yang dipelajari harus dirumuskan sesederhana mungkin. Kesederhanaan tidak harus berarti kehilangan informasi dan melewatkan bagian yang sulit.
Kedua, optimalkan susunan kata: Susunan kata pada item Anda harus dioptimalkan untuk memastikan bahwa dalam waktu singkat lampu yang tepat di otak Anda menyala. Hal ini akan mengurangi tingkat kesalahan, meningkatkan spesifisitas, mengurangi waktu respons, dan membantu konsentrasi Anda.

### Input:
Ciri-ciri Laut Mati: Danau garam yang terletak di perbatasan antara Israel dan Yordania. Garis pantainya merupakan titik terendah di permukaan bumi, rata-rata 396 m di bawah permukaan laut. Panjangnya 74 km. Tujuh kali lebih asin (30% volume) dibandingkan lautan. Kepadatannya membuat perenang tetap bertahan. Hanya organisme sederhana yang dapat hidup di perairan asinnya.

### Response:
{"Q": "Dimana letak Laut Mati?", "A": "di perbatasan antara Israel dan Yordania"}
{"Q": "Apa titik terendah di permukaan bumi?", "A": "Garis pantai Laut Mati"}
{"Q": "Berapakah ketinggian rata-rata di mana Laut Mati berada?", "A": "400 meter (di bawah permukaan laut)"}
{"Q": "Berapa panjang Laut Mati?", "A": "70 km"}
{"Q": "Seberapa asinkah Laut Mati dibandingkan dengan lautan?", "A": "7 kali"}
{"Q": "Berapa volume kandungan garam di Laut Mati?", "A": "30%"}
{"Q": "Mengapa Laut Mati bisa membuat para perenang tetap bertahan?", "A": "karena kandungan garamnya tinggi"}
{"Q": "Mengapa Laut Mati disebut Laut Mati?", "A": "karena hanya organisme sederhana yang dapat hidup di dalamnya"}
{"Q": "Mengapa hanya organisme sederhana yang dapat hidup di Laut Mati?", "A": "karena kandungan garamnya tinggi"}

### Input:
Sekolah menengah atas negeri dan swasta tersebar di berbagai wilayah di tanah air. SMA merupakan salah satu pilihan yang banyak dipilih pelajar selepas lulus SMP. Melanjutkan sekolah SMA lebih berpeluang untuk masuk ke perguruan tinggi negeri dengan memilih jurusan kuliah yang diminatinya.

Pemilihan jurusan bukan hanya dilakukan pada jenjang pendidikan pada SMK. Namun juga berlaku pada siswa siswi yang memilih masuk ke SMA. Jurusan yang terdapat pada SMA, antara lain: jurusan IPA, IPS dan jurusan bahasa.


### Response:
{"Q": "Apakah Sekolah Menegah Atas Negeri dan Swasta Tersebar Di Tanah Air?", "A": "Ya."}
{"Q": "Bagaimana Pemilihan Jurusan Dikelilingkingkan Pada Jenjang Pendidikan Pada SMK?", "A": "Hanya dilakukan pada jenjang pendidikan pada SMK."}
{"Q": "Bisa Juga Berlangsung Pada Siswa Siswi Yang Memilih Masuk Ke SMA?", "A": "Juga."}
{"Q": "Apakah Jurusan Yang Terdapat Pada SMA Antara Lain Jurusan IPA, IPS", "A": "Ya."}"""


def quantize_model():
    config = read_yaml_file("config/quantize_dummy.yaml")
    quantize_awq = QuantizeAWQ(config)
    quantize_awq.quantize()


def print_example_data():
    example_prompt = """Sekolah menengah atas negeri dan swasta tersebar di berbagai wilayah di tanah air. SMA merupakan salah satu pilihan yang banyak dipilih pelajar selepas lulus SMP. Melanjutkan sekolah SMA lebih berpeluang untuk masuk ke perguruan tinggi negeri dengan memilih jurusan kuliah yang diminatinya.

Pemilihan jurusan bukan hanya dilakukan pada jenjang pendidikan pada SMK. Namun juga berlaku pada siswa siswi yang memilih masuk ke SMA. Jurusan yang terdapat pada SMA, antara lain: jurusan IPA, IPS dan jurusan bahasa.
"""  # noqa: E50
    the_thing = GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE.format(
        input_user=example_prompt, response=""
    )
    print(the_thing)


def request_to_my_model():
    API_URL = "https://api-inference.huggingface.co/models/Erland/tinyllama-1.1B-chat-v0.3-dummy"
    API_TOKEN = os.getenv("HF_TOKEN_API")
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    data = query(
        {
            "inputs": EXAMPLE_INPUT,
            "options": {"wait_for_model": True}
            # "parameters": {"max_new_tokens": 250, "max_input_length": 1500},
        }
    )
    print(data)
    print(parse_output(data[0]["generated_text"]))


def example_tokenizer_count():
    count = TokenCounter.count("Erland/tinyllama-1.1B-chat-v0.3-dummy", EXAMPLE_INPUT)
    print(count)


def main() -> None:
    request_to_my_model()


if __name__ == "__main__":
    main()
