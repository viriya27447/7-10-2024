import streamlit as st

# เปลี่ยนสีพื้นหลังเป็นสีดำ
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: black;  /* สีพื้นหลังเป็นสีดำ */
        color: white;  /* เปลี่ยนสีข้อความเป็นสีขาว */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hello, Streamlit!")
st.write("นี่คือแอป Streamlit ที่มีพื้นหลังสีดำ")
st.write("ข้อความนี้จะมีสีขาวบนพื้นหลังสีดำ")
