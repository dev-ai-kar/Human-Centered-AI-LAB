import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("📄 Document summarization")
st.write(
    "Upload a document below and choose a summary format. GPT will summarize it! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Sidebar: Summary options and model selection
st.sidebar.header("⚙️ Options")

# Summary options
st.sidebar.subheader("📝 Summary Format")
summary_option = st.sidebar.radio(
    "Choose a summary format:",
    options=[
        "100 words",
        "2 connecting paragraphs",
        "5 bullet points"
    ],
    index=0
)

# Model selection
st.sidebar.subheader("🤖 Model Selection")
use_advanced_model = st.sidebar.checkbox(
    "Use advanced model",
    value=False,
    help="Check to use gpt-5-mini (advanced). Uncheck to use gpt-5-nano (basic)."
)

model_name = "gpt-5-mini" if use_advanced_model else "gpt-5-nano"
st.sidebar.markdown(f"**Selected Model:** `{model_name}`")

# Get OpenAI API key from Streamlit secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key not found in secrets. Please configure it in `.streamlit/secrets.toml`", icon="🗝️")
    st.stop()

if openai_api_key:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    if uploaded_file:

        # Process the uploaded file.
        document = uploaded_file.read().decode()

        # Build the prompt based on the selected summary format.
        if summary_option == "100 words":
            prompt = "Please summarize the following document in exactly 100 words."
        elif summary_option == "2 connecting paragraphs":
            prompt = "Please summarize the following document in 2 well-connected paragraphs."
        else:
            prompt = "Please summarize the following document in 5 bullet points."
        
        messages = [
            {
                "role": "user",
                "content": f"Here's a document:\n\n{document}\n\n---\n\n{prompt}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)
