import streamlit as st
import base64

st.set_page_config(
    page_title="Marketing Analytics App",
    page_icon="📊s",
    layout="wide",
)

st.write("# Welcome to the Marketing Analytics App! 📊")

# Read the image file
with open("logo.png", "rb") as image_file:
    bytes_data = image_file.read()

# Convert the bytes data to a data URL
data_url = base64.b64encode(bytes_data).decode()

# Use HTML and CSS to center the image
st.sidebar.markdown(
    f"""
    <div style="text-align: center">
        <img src="data:image/png;base64,{data_url}" width="150">
    </div>
    """,
    unsafe_allow_html=True,
)

# Add a blank line
st.sidebar.markdown("")

st.sidebar.success("Please select an option from the sidebar.")

st.markdown(
    """
    This application is designed for marketing consultants. It provides a suite of data science tools 
    that can be used to gain insights into marketing data. 
    **👈 Please select an option from the sidebar** to explore the features of this application.

    If you have any questions or need further assistance, feel free to contact me:

    Email: lukas.kostka@effectix.com
    """
)