# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# ---------------------
# Sidebar
# ---------------------
st.sidebar.title("📊 Stock Data Q&A App")
st.sidebar.write("Upload your stock CSV file and ask questions in natural language.")

# ---------------------
# File Upload
# ---------------------
uploaded_file = st.file_uploader("Upload your stock dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your dataset:")
    st.dataframe(df.head())

    # ---------------------
    # Setup LLM (replace with your API key)
    # ---------------------
    llm = OpenAI(api_token="XXX")  
    smart_df = SmartDataframe(df, config={"llm": llm})

    # ---------------------
    # User Query
    # ---------------------
    query = st.text_input("Ask a question about your stock data:")

    if query:
        with st.spinner("Fetching answer..."):
            try:
                response = smart_df.chat(query)

                # If response is text
                if isinstance(response, str):
                    st.write("### Answer:")
                    st.write(response)

                # If response is a dataframe (e.g., groupby, filter)
                elif isinstance(response, pd.DataFrame):
                    st.write("### DataFrame Result:")
                    st.dataframe(response)

                # If response is a matplotlib figure
                elif isinstance(response, plt.Figure):
                    st.write("### Visualization:")
                    st.pyplot(response)

                else:
                    st.write("### Raw Response:")
                    st.write(response)

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV file to get started.")
