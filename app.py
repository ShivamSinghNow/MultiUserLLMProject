# app.py

import streamlit as st
from streamlit_chat import message

from LocalLLMChat import MultiUserChatMemory

st.set_page_config(page_title="3-User Chat in VectorDB")

# Create a single ChatMemory instance for the entire app
if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = MultiUserChatMemory(model_name="qwen2.5")

# We'll track which user is selected
if "selected_user" not in st.session_state:
    st.session_state["selected_user"] = "Shivam"

# We'll keep the currently displayed messages in st.session_state["messages"]
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input text box
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""


def switch_user():
    """
    When the user picks a new radio button, load that user’s conversation
    from the DB, display it in st.session_state["messages"].
    """
    user = st.session_state["selected_user"]
    st.session_state["messages"] = st.session_state["chatbot"].load_user_conversation_as_tuples(user)

def display_messages():
    """
    Render the messages in st.session_state["messages"] with streamlit_chat.
    """
    st.subheader(f"Chat for {st.session_state['selected_user']}")
    for i, (txt, is_user) in enumerate(st.session_state["messages"]):
        message(txt, is_user=is_user, key=f"msg_{i}")

def process_input():
    """
    Called when the user presses Enter. We pass the new user message to the LLM,
    then re-load the conversation from the DB to display updated messages.
    """
    user_text = st.session_state["user_input"].strip()
    if not user_text:
        return

    user_id = st.session_state["selected_user"]
    # Ask the LLM
    with st.spinner("Thinking..."):
        _answer = st.session_state["chatbot"].ask(user_id, user_text)

    # Re-load conversation from DB
    st.session_state["messages"] = st.session_state["chatbot"].load_user_conversation_as_tuples(user_id)

    # Clear user input
    st.session_state["user_input"] = ""

def main():
    st.title("Three-User Chat with VectorDB Memory")

    # Radio button for user selection
    st.radio(
        "Select user",
        ["Shivam", "Johnathon", "Balthazar"],
        key="selected_user",
        on_change=switch_user
    )

    # Display current messages
    display_messages()

    # Chat input
    st.text_input(
        "Type your message",
        key="user_input",
        on_change=process_input
    )

    # Clear conversation for this user
    if st.button("Clear This User's Conversation"):
        current_user = st.session_state["selected_user"]
        st.session_state["chatbot"].clear_user_history(current_user)
        st.session_state["messages"] = []

if __name__ == "__main__":
    main()
