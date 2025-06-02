import streamlit as st
from app.chatbot import load_model, generate_response
from app.constants import sender_one, sender_two

def run_app():
    """
    Runs the Streamlit WhatsApp Chatbot UI.

    Displays a chat interface where the user can select a sender, enter messages,
    and view the conversation history. Handles message sending, response generation,
    and conversation reset.

    Returns:
        None
    """
    st.title("WhatsApp Chatbot - Simulate sender 1 or sender 2")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Map display name to internal speaker token format
    sender_display_name = st.selectbox("Choose your sender", [sender_one, sender_two])
    sender = f"<{sender_display_name}>"

    user_input = st.text_input("Your message:")

    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            st.session_state.chat_history.append((sender, user_input))
            bot_response = generate_response(st.session_state.chat_history, sender)
            other_sender = "<{sender_two}>".format(sender_two=sender_two) if sender == "<{sender_one}>".format(sender_one=sender_one) else "<{sender_one}>".format(sender_one=sender_one)
            st.session_state.chat_history.append((other_sender, bot_response))

    st.subheader("Conversation History")
    for speaker, message in st.session_state.chat_history:
        # Remove angle brackets for display
        display_name = speaker.replace("<", "").replace(">", "")
        st.write(f"**{display_name}:** {message}")

    if st.button("Reset Conversation"):
        st.session_state.chat_history = []
        st.rerun()
