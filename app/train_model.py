# wwtt/app/train_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch


def load_and_preprocess_data(data_path):
    """
    Loads and preprocesses chat data for model training.

    Reads a CSV file containing chat messages, groups messages by conversation,
    chunks them to a maximum word count, and prepares them for language model training.
    Also adds special speaker tokens to the tokenizer.

    Args:
        data_path (str): Path to the CSV file with columns ['message', 'sender', 'conversation_id'].

    Returns:
        tuple: (tokenized_dataset, tokenizer)
            tokenized_dataset: HuggingFace Dataset ready for training.
            tokenizer: The tokenizer with added speaker tokens.
    """
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["message", "sender", "conversation_id"])

    # Normalize sender names
    df['sender'] = df['sender'].str.strip()

    grouped = []
    max_words_per_chunk = 150  # guard for chunk size

    # Group messages by conversation_id
    for conv_id, conv_df in df.groupby("conversation_id"):
        current_chunk = ""
        for _, row in conv_df.iterrows():
            sender = row['sender']
            message = str(row['message']).strip()
            line = f"<{sender}>: {message}"

            # Add newline between messages
            if current_chunk:
                current_chunk += "\n"
            current_chunk += line

            if len(current_chunk.split()) > max_words_per_chunk:
                grouped.append({"text_for_model": current_chunk.strip()})
                current_chunk = ""

        if current_chunk.strip():
            grouped.append({"text_for_model": current_chunk.strip()})

    dataset = Dataset.from_list(grouped)

    # Load tokenizer and add special speaker tokens
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    unique_speakers = list(df["sender"].dropna().unique())
    speaker_tokens = [f"<{name}>" for name in unique_speakers]
    tokenizer.add_special_tokens({'additional_special_tokens': speaker_tokens})

    def tokenize_function(examples):
        """
        Tokenizes the input text for the model.

        Args:
            examples (dict): Dictionary with key 'text_for_model'.

        Returns:
            dict: Tokenized inputs with labels.
        """
        texts = examples["text_for_model"]
        if isinstance(texts, str):
            texts = [texts]
        tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset, tokenizer

def train_and_save_model():
    """
    Trains a causal language model on the preprocessed chat data and saves the model and tokenizer.

    Loads the data, splits into train/test, trains the model, and saves the results to disk.
    """
    data_path = "./data/processed/cleaned_chat.csv"
    tokenized_dataset, tokenizer = load_and_preprocess_data(data_path)

    train_test = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./data/model",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model("./data/model")
    tokenizer.save_pretrained("./data/model")
    print("Model and tokenizer saved to ./data/model")

if __name__ == "__main__":
    train_and_save_model()
