import streamlit as st

def page1():
    st.write('hello1')

def page2():
    st.write('hello2')

pg = st.navigation([st.Page(page1), st.Page(page2)])
pg.run()