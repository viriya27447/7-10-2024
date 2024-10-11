import streamlit as st

# กำหนดค่าเริ่มต้นใน session state
if 'foo' not in st.session_state:
    st.session_state.foo = "A"  # ค่าเริ่มต้น
if 'bar' not in st.session_state:
    st.session_state.bar = False  # ค่าเริ่มต้น

# ฟังก์ชันสำหรับหน้า 1
def page1():
    st.title("หน้า 1")
    st.write("ค่าที่เลือกใน Foo:", st.session_state.foo)

# ฟังก์ชันสำหรับหน้า 2
def page2():
    st.title("หน้า 2")
    st.write("สถานะของ Bar:", "เปิด" if st.session_state.bar else "ปิด")

# Sidebar สำหรับการเลือกและการจัดการค่า
st.sidebar.title("เมนู")
st.sidebar.selectbox("เลือกค่า Foo:", ["A", "B", "C"], key="foo")
st.sidebar.checkbox("Bar", key="bar")

# การเลือกหน้า
page = st.sidebar.radio("เลือกหน้า:", ["หน้า 1", "หน้า 2"])

# แสดงเนื้อหาตามหน้า
if page == "หน้า 1":
    page1()
else:
    page2()
