import streamlit as st

# ตั้งค่าหน้า
st.set_page_config(
    page_title="My Streamlit App",
    page_icon="🌟",
    layout="wide",  # หรือ "centered" สำหรับการจัดเรียงเนื้อหา
)

st.title("Hello, Streamlit!")
st.write("นี่คือแอป Streamlit ที่ไม่มีการใช้ CSS")
