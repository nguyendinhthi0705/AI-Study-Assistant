import streamlit as st 
import Libs as glib 
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Search Knowledge base")
input_text = st.text_input("Search Knowledge base") 
if input_text: 
    st_callback = StreamlitCallbackHandler(st.container())
    response = glib.search_new(input_text)
    full_response = st.write_stream(response)
  
    
