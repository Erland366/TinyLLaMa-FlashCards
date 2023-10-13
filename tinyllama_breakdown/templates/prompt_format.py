ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{user}\n\n### Response:\n{assistant} "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{assistant} "
    ),
}
ALPACA_ID_PROMPT_DICT = {
    "prompt_input": (
        "Berikut adalah instruksi yang menjelaskan sebuah tugas, dipasangkan dengan masukan yang memberikan konteks lebih lanjut. "
        "Tulis respons yang secara tepat memenuhi permintaan.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{user}\n\n### Response:\n{assistant} "
    ),
    "prompt_no_input": (
        "Berikut adalah instruksi yang menjelaskan tugas. "
        "Tulis respons yang secara tepat memenuhi permintaan.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{assistant} "
    ),
}

CHATML_ID_PROMPT_DICT = {
    "prompt_input": (
        "Berikut adalah instruksi yang menjelaskan sebuah tugas, dipasangkan dengan masukan yang memberikan konteks lebih lanjut. "
        "Tulis respons yang secara tepat memenuhi permintaan.\n\n"
        "<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\
\n<|im_start|>assistant\n{assistant}<|im_end|>\n"
    ),
    "prompt_no_input": (
        "Berikut adalah instruksi yang menjelaskan tugas. "
        "Tulis respons yang secara tepat memenuhi permintaan.\n\n"
        "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}\
<|im_end|>\n"
    ),
}

ANKI_CARDS_ID_PROMPT = """Saya ingin Anda berperan sebagai pembuat Anki Cards profesional, mampu membuat Anki Cards dari teks yang saya berikan.

Mengenai perumusan isi kartu, Anda berpegang pada dua prinsip:
Pertama, prinsip informasi minimum: Materi yang dipelajari harus dirumuskan sesederhana mungkin. Kesederhanaan tidak harus berarti kehilangan informasi dan melewatkan bagian yang sulit.
Kedua, optimalkan susunan kata: Susunan kata pada item Anda harus dioptimalkan untuk memastikan bahwa dalam waktu singkat lampu yang tepat di otak Anda menyala. Hal ini akan mengurangi tingkat kesalahan, meningkatkan spesifisitas, mengurangi waktu respons, dan membantu konsentrasi Anda.

Berikut ini adalah contoh template pembuatan kartu untuk Anda pelajari.

Teks: Ciri-ciri Laut Mati: Danau garam yang terletak di perbatasan antara Israel dan Yordania. Garis pantainya merupakan titik terendah di permukaan bumi, rata-rata 396 m di bawah permukaan laut. Panjangnya 74 km. Tujuh kali lebih asin (30% volume) dibandingkan lautan. Kepadatannya membuat perenang tetap bertahan. Hanya organisme sederhana yang dapat hidup di perairan asinnya.

Buatlah kartu berdasarkan teks di atas sebagai berikut:

T: Dimana letak Laut Mati? Jawaban: di perbatasan antara Israel dan Yordania
Q: Apa titik terendah di permukaan bumi? A: Garis pantai Laut Mati
T: Berapakah ketinggian rata-rata di mana Laut Mati berada? A: 400 meter (di bawah permukaan laut)
Q: Berapa panjang Laut Mati? J: 70 km
T: Seberapa asinkah Laut Mati dibandingkan dengan lautan? J: 7 kali
Q: Berapa volume kandungan garam di Laut Mati? J: 30%
T: Mengapa Laut Mati bisa membuat para perenang tetap bertahan? A: karena kandungan garamnya tinggi
Q: Mengapa Laut Mati disebut Laut Mati? A: karena hanya organisme sederhana yang dapat hidup di dalamnya
T: Mengapa hanya organisme sederhana yang dapat hidup di Laut Mati? A: karena kandungan garamnya tinggi
"""

GENERATE_DATA_ANKI_CARDS_MD_TEMPLATE = """### Instruction:
Saya ingin Anda berperan sebagai pembuat Anki Cards profesional, mampu membuat Anki Cards dari teks yang saya berikan.

Mengenai perumusan isi kartu, Anda berpegang pada dua prinsip:
Pertama, prinsip informasi minimum: Materi yang dipelajari harus dirumuskan sesederhana mungkin. Kesederhanaan tidak harus berarti kehilangan informasi dan melewatkan bagian yang sulit.
Kedua, optimalkan susunan kata: Susunan kata pada item Anda harus dioptimalkan untuk memastikan bahwa dalam waktu singkat lampu yang tepat di otak Anda menyala. Hal ini akan mengurangi tingkat kesalahan, meningkatkan spesifisitas, mengurangi waktu respons, dan membantu konsentrasi Anda.

### Input:
Ciri-ciri Laut Mati: Danau garam yang terletak di perbatasan antara Israel dan Yordania. Garis pantainya merupakan titik terendah di permukaan bumi, rata-rata 396 m di bawah permukaan laut. Panjangnya 74 km. Tujuh kali lebih asin (30% volume) dibandingkan lautan. Kepadatannya membuat perenang tetap bertahan. Hanya organisme sederhana yang dapat hidup di perairan asinnya.

### Response:
T: Dimana letak Laut Mati? Jawaban: di perbatasan antara Israel dan Yordania
Q: Apa titik terendah di permukaan bumi? A: Garis pantai Laut Mati
T: Berapakah ketinggian rata-rata di mana Laut Mati berada? A: 400 meter (di bawah permukaan laut)
Q: Berapa panjang Laut Mati? J: 70 km
T: Seberapa asinkah Laut Mati dibandingkan dengan lautan? J: 7 kali
Q: Berapa volume kandungan garam di Laut Mati? J: 30%
T: Mengapa Laut Mati bisa membuat para perenang tetap bertahan? A: karena kandungan garamnya tinggi
Q: Mengapa Laut Mati disebut Laut Mati? A: karena hanya organisme sederhana yang dapat hidup di dalamnya
T: Mengapa hanya organisme sederhana yang dapat hidup di Laut Mati? A: karena kandungan garamnya tinggi

### Input:
{input}

### Response:
{response}
"""

GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE = """### Instruction:
Saya ingin Anda berperan sebagai pembuat Anki Cards profesional, mampu membuat Anki Cards dari teks yang saya berikan.

Mengenai perumusan isi kartu, Anda berpegang pada dua prinsip:
Pertama, prinsip informasi minimum: Materi yang dipelajari harus dirumuskan sesederhana mungkin. Kesederhanaan tidak harus berarti kehilangan informasi dan melewatkan bagian yang sulit.
Kedua, optimalkan susunan kata: Susunan kata pada item Anda harus dioptimalkan untuk memastikan bahwa dalam waktu singkat lampu yang tepat di otak Anda menyala. Hal ini akan mengurangi tingkat kesalahan, meningkatkan spesifisitas, mengurangi waktu respons, dan membantu konsentrasi Anda.

### Input:
Ciri-ciri Laut Mati: Danau garam yang terletak di perbatasan antara Israel dan Yordania. Garis pantainya merupakan titik terendah di permukaan bumi, rata-rata 396 m di bawah permukaan laut. Panjangnya 74 km. Tujuh kali lebih asin (30% volume) dibandingkan lautan. Kepadatannya membuat perenang tetap bertahan. Hanya organisme sederhana yang dapat hidup di perairan asinnya.

### Response:
{{"Q": "Dimana letak Laut Mati?", "A": "di perbatasan antara Israel dan Yordania"}}
{{"Q": "Apa titik terendah di permukaan bumi?", "A": "Garis pantai Laut Mati"}}
{{"Q": "Berapakah ketinggian rata-rata di mana Laut Mati berada?", "A": "400 meter (di bawah permukaan laut)"}}
{{"Q": "Berapa panjang Laut Mati?", "A": "70 km"}}

### Input:
{input_user}

### Response:
{response}"""

GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE_EN = """### Instructions:
I want you to act as a professional Anki Cards maker, able to create Anki Cards from the text I provide.

Regarding the formulation of the contents of the card, you adhere to two principles:
First, the principle of minimum information: The material studied should be formulated as simply as possible. Simplicity doesn't have to mean losing information and skipping difficult parts.
Second, optimize the wording: The wording of your items should be optimized to ensure that in no time the right lights in your brain turn on. This will reduce error rates, increase specificity, reduce response times, and help your concentration.

### Inputs:
Characteristics of the Dead Sea: Salt lake located on the border between Israel and Jordan. The coastline is the lowest point on the earth's surface, an average of 396 m below sea level. Its length is 74 km. Seven times saltier (30% by volume) than the ocean. Its density keeps swimmers afloat. Only simple organisms can live in its salty waters.

### Response:
{{"Q": "Where is the Dead Sea?", "A": "on the border between Israel and Jordan"}}
{{"Q": "What is the lowest point on the earth's surface?", "A": "Dead Sea coastline"}}
{{"Q": "What is the average height at which the Dead Sea is located?", "A": "400 meters (below sea level)"}}
{{"Q": "How long is the Dead Sea?", "A": "70 km"}}

### Inputs:
{user_input}

### Response:
{response}"""

OTHER_EXAMPLES = """
{{"Q": "Seberapa asinkah Laut Mati dibandingkan dengan lautan?", "A": "7 kali"}}
{{"Q": "Berapa volume kandungan garam di Laut Mati?", "A": "30%"}}
{{"Q": "Mengapa Laut Mati bisa membuat para perenang tetap bertahan?", "A": "karena kandungan garamnya tinggi"}}
{{"Q": "Mengapa Laut Mati disebut Laut Mati?", "A": "karena hanya organisme sederhana yang dapat hidup di dalamnya"}}
{{"Q": "Mengapa hanya organisme sederhana yang dapat hidup di Laut Mati?", "A": "karena kandungan garamnya tinggi"}}
"""

CHATML_TEMPLATE = """<|im_start|>user:{input_user}<|im_end|>\n<|im_start|>assistant:"""
