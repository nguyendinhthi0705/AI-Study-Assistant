import streamlit as st 
import Libs as glib 
import json

st.set_page_config(page_title="Home")

st.markdown("Top 10 interview questions for OOP in Java") 
st.markdown("Phân biệt giữa classifcation and object detection trong computer vision.") 

input_text = st.text_area("Input your question") 
if input_text: 
    with st.chat_message("user"): 
        st.markdown(input_text) 
    response = glib.call_claude_sonet_stream(input_text)
    st.write_stream(response)

    



    
   