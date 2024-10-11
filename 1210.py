import streamlit as st

# ตั้งชื่อแอป
st.title("Remove BG - AI Background Removal Tool")

# กำหนด HTML และ CSS
html_content = """
<div style="font-family: Arial, sans-serif; background-color: #1E1E1E; color: white; margin: 0; padding: 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; height: 100vh; padding: 0 50px;">
        <div style="width: 30%; background: #333; padding: 20px; border-radius: 10px; text-align: center;">
            <img src="https://firebasestorage.googleapis.com/v0/b/project-5195649815793865937.appspot.com/o/Monica_2024-10-12_00-56-14.png?alt=media&token=8060b05e-9406-4907-ba64-aa2136e72cbe" style="width: 100%; max-height: 250px; margin-bottom: 20px; border: 2px dashed #F20559; border-radius: 10px;" alt="image preview">
            <button style="background: #F20559; color: white; border: none; padding: 15px 30px; cursor: pointer; border-radius: 5px;">เลือกรูปภาพ</button>
        </div>
        <div style="width: 60%;">
            <h1 style="font-size: 2.5rem; margin-bottom: 10px;">เครื่องมือลบพื้นหลังรูปภาพฟรีด้วย AI</h1>
            <div style="color: #F20559; font-size: 1.2rem; margin-bottom: 20px;">★★★★★</div>
            <p style="font-size: 1.1rem; margin-bottom: 30px;">ลบพื้นหลังฟรีและอัตโนมัติ 100%! เพียงคลิกเดียวก็ลบพื้นหลังได้ในไม่กี่วินาที!</p>
        </div>
    </div>
</div>
"""

# แสดง HTML ใน Streamlit
st.markdown(html_content, unsafe_allow_html=True)
