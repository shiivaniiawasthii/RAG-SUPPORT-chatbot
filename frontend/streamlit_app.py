import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Support Doc Chatbot", page_icon="💬")
st.title("Support Doc Chatbot")
st.caption("Upload a document, then ask questions about it.")

if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF, TXT, or MD file", type=["pdf", "txt", "md"])

if uploaded_file and st.button("Process document"):
    with st.spinner("Reading and indexing document..."):
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
        )
    if response.status_code == 200:
        data = response.json()
        st.session_state.document_id = data["document_id"]
        st.session_state.messages = []
        st.success(f"Indexed {data['chunk_count']} chunks from {data['filename']}")
    else:
        st.error(f"Upload failed: {response.json().get('detail', response.text)}")

if st.session_state.document_id:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about the document")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"document_id": st.session_state.document_id, "question": question},
            )

        with st.chat_message("assistant"):
            if response.status_code == 200:
                data = response.json()
                st.write(data["answer"])
                with st.expander("Sources"):
                    for source in data["sources"]:
                        st.markdown(f"- {source['text'][:300]}...")
                st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
            else:
                error = response.json().get("detail", response.text)
                st.error(error)
else:
    st.info("Upload and process a document to start chatting.")
