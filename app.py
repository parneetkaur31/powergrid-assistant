import streamlit as st

st.set_page_config(page_title="PowerGrid Assistant", page_icon="⚡")

st.title("⚡ PowerGrid Assistant")

st.write(
    "PowerGrid Assistant is an AI chatbot designed to help engineers query technical "
    "documentation related to Substation Automation Systems and power grid infrastructure."
)

query = st.text_input("Ask a question about the power grid documentation:")

if query:
    st.write("🔎 Searching knowledge base...")
    st.success("Answer generation pipeline will appear here.")