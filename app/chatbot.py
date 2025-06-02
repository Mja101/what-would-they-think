import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import emoji
from app.constants import sender_one, sender_two, model_path

@st.cache_resource
def load_model():
    """
    Loads the tokenizer and model from the specified local directory.
    Uses Streamlit's resource cache to avoid reloading on every run.
    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

def generate_response(chat_history, current_sender):
    """
    Generates a chatbot response based on the chat history and the current sender.
    Constructs a prompt from the chat history, generates a response using the model,
    and extracts the reply for the other participant.
    Args:
        chat_history (list): List of (speaker, message) tuples.
        current_sender (str): Name of the user sending the message.
    Returns:
        str: The generated response for the other participant, with emojis rendered.
    """
    tokenizer, model = load_model()

    prompt_text = ""
    for speaker, message in chat_history:
        prompt_text += f"{speaker}: {message}\n"

    # Switch between full names
    other_sender = sender_two if current_sender == sender_one else sender_one
    prompt_text += f"{other_sender}:"

    inputs = tokenizer.encode(prompt_text, return_tensors='pt')

    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 50,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_p=0.95,
        top_k=50
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_lines = response_text.split('\n')

    bot_response = ""
    for line in response_lines[::-1]:
        if line.startswith(f"{other_sender}:"):
            bot_response = line[len(f"{other_sender}:"):].strip()
            break
        
    # Convert emoji shortcodes (like :smile:) into actual emojis
    bot_response = emoji.emojize(bot_response, language='alias')

    return bot_response
