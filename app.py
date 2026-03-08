import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(page_title="PowerGrid Assistant")

st.title("⚡ PowerGrid Assistant")
st.write("Ask questions about the power grid documentation")

# store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
query = st.chat_input("Ask a question")

if query:

    # show user message
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    # generate answer
    with st.spinner("Searching documents..."):
        answer, sources = ask_question(query)

    response = answer

    if sources:
        response += "\n\n**Sources:**\n"
        for s in sources:
            response += f"- {s}\n"

    # show assistant message
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )