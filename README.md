# What Would They Think (WWTT)

A project for parsing WhatsApp chat exports, training a conversational language model on your own chats, and chatting with the model in a Streamlit web app.

---

## Features

- **Parse WhatsApp chat exports** into structured CSV data.
- **Train a GPT-2-based conversational model** on your own WhatsApp conversations.
- **Chat with your trained model** using a simple Streamlit UI.

---

## Setup

1. **Clone the repository**
    ```sh
    git clone https://github.com/yourusername/wwtt.git
    cd wwtt
    ```

2. **Create and activate a virtual environment**
    ```sh
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

---

## 1. Parse WhatsApp Chat Export

Export your WhatsApp chat as a `.txt` file (from WhatsApp, choose "Export chat" without media).

Then run the parsing script:

```sh
python -m scripts.parse_whatsapp --input path/to/your_chat.txt --output data/processed/chat.csv
```

- `--input`: Path to your exported WhatsApp `.txt` file.
- `--output`: Path to save the parsed CSV file.

---

## 2. Train the Model

Make sure your parsed CSV is at `data/processed/chat.csv` (or update the path in `app/train_model.py`).

Then train the model:

```sh
python -m app.train_model
```

- The script will preprocess the data, train a GPT-2 model, and save the model and tokenizer to `data/model/`.

---

## 3. Launch the Chatbot App

Start the Streamlit app:

```sh
streamlit run streamlit_app.py
```

- Open the provided local URL in your browser.
- Choose a sender, type a message, and chat with your trained model!

### Example Chats

An example conversation, this is pretty rudimentary stuff but cool to see in action. This was trained from my partner and I's conversations.

![alt text](<assets/Examples/Screenshot 2025-06-16 135226.png>)

---



## File Structure

```
wwtt/
│
├── app/
│   ├── chatbot.py              # Chatbot logic and model loading
│   ├── data_utils.py           # WhatsApp parsing and CSV utilities
│   ├── train_model.py          # Model training script
│   └── ui.py                   # Streamlit UI logic
│
├── data/
│   ├── processed/              # Place for parsed CSV files
│   └── model/                  # Saved model and tokenizer
│
├── scripts/
│   └── parse_whatsapp.py       # WhatsApp .txt to .csv parser
|
├── tests/
│   └── test_chatbot.py         # Unit tests for chatbot.py
│   └── test_data_utils.py      # Unit tests for data_utils.py
│   └── test_train_model.py     # Unit tests for train_model.py
|
├── streamlit_app.py            # Streamlit app entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Notes

- The model is based on [GPT-2](https://huggingface.co/gpt2) and fine-tuned on your chat data.
- Only text messages are supported; media messages are ignored.
- For best results, use chats with two participants.

---

## To do

- Implement sqlite to store conversations (locally)

---

## License

MIT License

---