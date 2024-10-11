import streamlit as st

# เปลี่ยนสีพื้นหลังเป็นสีดำ
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: black;  /* สีพื้นหลังเป็นสีดำ */
        color: white;  /* เปลี่ยนสีข้อความเป็นสีขาวเพื่อให้อ่านได้ชัดเจน */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hello, Streamlit!")
st.write("นี่คือแอป Streamlit ที่มีพื้นหลังสีดำ")
