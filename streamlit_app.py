"""Streamlit front end for the Kartify order chatbot.

Run locally:  streamlit run streamlit_app.py
On Streamlit Cloud set secrets OPENAI_API_KEY (and optionally OPENAI_API_BASE, TYPESAFE_API_KEY).
"""
import os
import sqlite3

import pandas as pd
import streamlit as st

# Export Streamlit secrets to env vars *before* importing agent (it builds the LLMs on import).
try:
    for key in ("OPENAI_API_KEY", "OPENAI_API_BASE", "TYPESAFE_API_KEY"):
        if key in st.secrets and st.secrets[key]:
            os.environ["OPENAI_BASE_URL" if key == "OPENAI_API_BASE" else key] = st.secrets[key]
except Exception:
    pass  # no secrets file (local run): agent.py falls back to config.json

st.set_page_config(page_title="Kartify Support", page_icon="🛒")

if not os.environ.get("OPENAI_API_KEY") and not os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
):
    st.error("OPENAI_API_KEY is not set. Add it under the app's Settings → Secrets.")
    st.stop()

import agent  # noqa: E402

st.title("🛒 Kartify Order Support")

# ---- Step 1: identify customer and order ----
if "state" not in st.session_state:
    cust_id = st.text_input("Customer ID", placeholder="e.g. C1014").strip().upper()
    if cust_id:
        with sqlite3.connect(agent.DB_PATH) as conn:
            orders = pd.read_sql_query(
                "SELECT order_id, product_description FROM orders WHERE customer_id = ?",
                conn,
                params=(cust_id,),
            )
        if orders.empty:
            st.warning(f"No orders found for customer {cust_id}.")
        else:
            st.dataframe(orders, hide_index=True)
            order_id = st.selectbox("Order", orders["order_id"])
            if st.button("Start chat"):
                st.session_state.state = agent.new_state(cust_id, order_id)
                st.session_state.messages = []
                st.session_state.ended = False
                st.rerun()
    st.stop()

# ---- Step 2: chat ----
state = st.session_state.state
st.caption(f"Customer {state['cust_id']} · Order {state['order_id']}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if st.session_state.ended:
    st.info("This conversation has ended.")
    if st.button("Start over"):
        del st.session_state["state"]
        st.rerun()
    st.stop()

if query := st.chat_input("Ask about your order"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    state["query"] = query
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                state, ended = agent.process_query(state)
            except Exception as e:
                st.error(f"Sorry, something went wrong: {type(e).__name__}: {e}")
                st.session_state.messages.pop()  # let the user retry the same question
                st.stop()
        st.write(state["final_response"])
    st.session_state.state = state
    st.session_state.ended = ended
    st.session_state.messages.append({"role": "assistant", "content": state["final_response"]})
    if ended:
        st.rerun()
