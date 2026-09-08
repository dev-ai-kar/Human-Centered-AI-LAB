import streamlit as st
import tiktoken
from openai import OpenAI

# Show title and description.
st.title("Conversational AI")
st.write("Ask questions and get answers from GPT.")

model_to_use = st.sidebar.selectbox(
    "Which model?", ("gpt-4o-mini", "gpt-4o"), index=0
)
max_tokens = st.sidebar.number_input(
    "Maximum context tokens", min_value=256, max_value=128000, value=4000, step=256
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def count_tokens(messages, model):
    """Count the tokens in a chat-completions message list."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    return sum(
        4 + len(encoding.encode(message["content"])) for message in messages
    ) + 2


def conversation_buffer(messages, token_limit, model):
    """Return the newest complete two-turn history within the token limit."""
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    buffered_messages = [system_message]
    history = messages[-4:]

    for message in history:
        candidate = buffered_messages + [message]
        if count_tokens(candidate, model) > token_limit:
            continue
        buffered_messages.append(message)

    return buffered_messages

# Get the OpenAI API key from Streamlit secrets.
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error(
        "OpenAI API key not found in secrets. Please configure it in "
        "`.streamlit/secrets.toml`",
        icon="🗝️",
    )
    st.stop()

if openai_api_key:
    client = OpenAI(api_key=openai_api_key)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        messages_for_request = conversation_buffer(
            st.session_state.messages, max_tokens, model_to_use
        )
        request_tokens = count_tokens(messages_for_request, model_to_use)
        st.caption(f"Request context: {request_tokens:,} tokens")

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages_for_request,
                stream=True,
            )
            response = st.write_stream(stream)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
