import streamlit as st
import streamlit.components.v1 as components

# สร้างฟังก์ชันเพื่อแสดงพื้นหลังสีดำ
def set_black_background():
    components.html(
        """
        <style>
        body {
            background-color: black;
            color: white; /* เปลี่ยนสีข้อความเป็นสีขาว */
        }
        </style>
        """,
        height=0,  # กำหนดความสูงเป็น 0 เพื่อไม่ให้มีส่วนแสดงผล
    )

# เรียกใช้ฟังก์ชัน
set_black_background()

st.title("Hello, Streamlit!")
st.write("นี่คือแอป Streamlit ที่มีพื้นหลังสีดำโดยใช้ไลบรารี components")
