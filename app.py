import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(page_title="PowerGrid Assistant")

st.title("⚡ PowerGrid Assistant")

st.write("Ask questions about the power grid documentation")

query = st.text_input("Ask a question")

if query:

    with st.spinner("Searching documents..."):

        answer, sources = ask_question(query)

    st.write(answer)

    if sources:
        st.markdown("### Sources")
        for s in sources:
            st.write(s)