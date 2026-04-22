from __future__ import annotations

import asyncio

import streamlit as st

from models import ErrorInput
from decision import explain_error
from llm import get_llm_client


async def run_explanation(error_message: str, code_snippet: str):
    client = get_llm_client()
    sample = ErrorInput(
        error_message=error_message,
        code_snippet=code_snippet,
    )
    return await explain_error(sample, client)


st.set_page_config(page_title="Program Error Explanation Agent", page_icon="🤖")

st.title("Program Error Explanation Agent")
st.write("Paste an error message and a short code snippet to get a beginner-friendly explanation.")

error_message = st.text_input(
    "Error message",
    placeholder="Example: TypeError: unsupported operand type(s) for +: 'int' and 'str'",
)

code_snippet = st.text_area(
    "Code snippet",
    height=220,
    placeholder="""Example:
age = input("Enter your age: ")
years_until_100 = 100 + age
print(f"You have {years_until_100} years until you turn 100!")""",
)

if st.button("Explain Error"):
    if not error_message.strip() or not code_snippet.strip():
        st.warning("Please provide both an error message and a code snippet.")
    else:
        with st.spinner("Analyzing error..."):
            try:
                result = asyncio.run(run_explanation(error_message, code_snippet))

                st.subheader("Explanation")
                st.markdown(f"**Error Type:**  \n{result.error_type}")
                st.markdown(f"**What happened:**  \n{result.plain_explanation}")
                st.markdown(f"**Most likely cause:**  \n{result.likely_cause}")
                st.markdown(f"---")
                st.markdown("**Debugging steps:**")
                for step in result.debug_steps:
                    st.markdown(f"- {step}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")