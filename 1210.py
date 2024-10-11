import streamlit as st

def page1():
    st.write(st.session_state.foo)

def page2():
    st.write(st.session_state.bar)

pg = st.navigation([st.Page(page1), st.Page(page2)])
pg.run()