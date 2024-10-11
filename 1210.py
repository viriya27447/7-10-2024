import streamlit as st
from PIL import Image

# ตั้งค่าหน้าแอป
st.set_page_config(page_title="Remove BG - AI Background Removal Tool", layout="wide")

# โครงสร้าง HTML
html_content = """
<div style="font-family: Arial, sans-serif; background-color: #1E1E1E; color: white; margin: 0; padding: 0; height: 100vh; display: flex; justify-content: center; align-items: center;">
    <div style="display: flex; width: 100%; max-width: 1200px; padding: 20px;">
        <div style="flex: 1; background: #333; padding: 20px; border-radius: 10px; text-align: center; margin-right: 20px;">
            <img src="https://via.placeholder.com/300" id="image-preview" style="width: 100%; max-height: 250px; margin-bottom: 20px; border: 2px dashed #F20559; border-radius: 10px;" alt="image preview">
            <button style="background: #F20559; color: white; border: none; padding: 15px 30px; cursor: pointer; border-radius: 5px;">เลือกรูปภาพ</button>
        </div>
        <div style="flex: 2; padding: 20px;">
            <h1 style="font-size: 2.5rem; margin-bottom: 10px;">เครื่องมือลบพื้นหลังรูปภาพฟรีด้วย AI</h1>
            <div style="color: #F20559; font-size: 1.2rem; margin-bottom: 20px;">★★★★★</div>
            <p style="font-size: 1.1rem; margin-bottom: 30px;">ลบพื้นหลังฟรีและอัตโนมัติ 100%! เพียงคลิกเดียวก็ลบพื้นหลังได้ในไม่กี่วินาที!</p>
        </div>
    </div>
</div>
"""

# แสดง HTML ใน Streamlit
st.markdown(html_content, unsafe_allow_html=True)

# ฟังก์ชันการอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption='ภาพที่อัปโหลด', use_column_width=True)
