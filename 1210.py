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

# Sidebar สำหรับการเลือกค่า
st.sidebar.title("เมนู")
st.sidebar.selectbox("เลือกค่า Foo:", ["A", "B", "C"], key="foo")
st.sidebar.checkbox("Bar", key="bar")

# ปุ่มสำหรับการเปลี่ยนหน้า
if st.button("ไปที่หน้า 1"):
    st.session_state.current_page = "page1"

if st.button("ไปที่หน้า 2"):
    st.session_state.current_page = "page2"

# กำหนดค่าเริ่มต้นของหน้าเมื่อเริ่มต้น
if 'current_page' not in st.session_state:
    st.session_state.current_page = "page1"

# แสดงเนื้อหาตามหน้า
if st.session_state.current_page == "page1":
    page1()
else:
    page2()
