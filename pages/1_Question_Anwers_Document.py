import streamlit as st 
import Libs as glib 
from PyPDF2 import PdfReader
import Libs as glib 


st.set_page_config(page_title="Questions on Lecture")

uploaded_file = st.file_uploader("Upload Your Lecture in PDF")
docs = []

st.markdown("Ask me anything as below samples:") 
st.markdown("Summary the financial report of Amazon") 
st.markdown("Thẻ có những chức năng gì") 

input_text = st.text_input("Your question!") 
if uploaded_file is not None and input_text:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        docs.append(page.extract_text())
    
    response = glib.query_document(input_text, docs)
    st.write_stream(response)


   