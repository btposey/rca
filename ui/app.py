import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8080")

st.title("Restaurant Concierge")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = ""

with st.sidebar:
    if st.button("Start over"):
        st.session_state.messages = []
        st.session_state.context = ""
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What are you looking for?"):
    combined = f"{st.session_state.context}, {prompt}" if st.session_state.context else prompt

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding your restaurant..."):
            try:
                resp = requests.post(f"{API_URL}/query", json={"query": combined}, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                if data.get("attack"):
                    reply = data["suggestion"]
                    st.session_state.context = ""
                elif data.get("suggestion", "").startswith("I don't have any"):
                    reply = data["suggestion"]
                    st.session_state.context = ""
                else:
                    reply = data["suggestion"]
                    if data.get("elaboration"):
                        reply += f"\n\n{data['elaboration']}"
                    st.session_state.context = combined

            except requests.exceptions.RequestException as e:
                reply = f"Could not reach the API: {e}"
                st.session_state.context = ""

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
