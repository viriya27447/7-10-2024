import streamlit as st

# สร้างฟังก์ชันสำหรับแต่ละหน้า
def page_one():
    st.title("หน้า 1")
    st.write("ยินดีต้อนรับสู่หน้า 1!")

def page_two():
    st.title("หน้า 2")
    st.write("ยินดีต้อนรับสู่หน้า 2!")

# เมนูด้านข้าง
st.sidebar.title("เมนู")
page = st.sidebar.radio("เลือกหน้า:", ("หน้า 1", "หน้า 2"))

# แสดงหน้าตามที่เลือก
if page == "หน้า 1":
    page_one()
else:
    page_two()
