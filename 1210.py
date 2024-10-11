import streamlit as st
from PIL import Image

# ตั้งค่าหน้าแอป
st.set_page_config(page_title="Remove BG - AI Background Removal Tool", layout="wide")

# เปลี่ยนพื้นหลังด้วย st.write
st.write(
    """
    <style>
    body {
        background-color: #1E1E1E;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# แบ่งคอลัมน์
col1, col2 = st.columns([3, 7])  # คอลัมน์ซ้าย 30% ขวา 70%

# ส่วนสำหรับอัปโหลดรูปภาพ
with col1:
    st.header("เลือกรูปภาพ")
    uploaded_file = st.file_uploader("อัปโหลดไฟล์รูปภาพ", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='ภาพที่อัปโหลด', use_column_width=True)

# ส่วนสำหรับข้อความ
with col2:
    st.title("เครื่องมือลบพื้นหลังรูปภาพฟรีด้วย AI")
    st.markdown("⭐️⭐️⭐️⭐️⭐️")
    st.write("ลบพื้นหลังฟรีและอัตโนมัติ 100%! เพียงคลิกเดียวก็ลบพื้นหลังได้ในไม่กี่วินาที!")
