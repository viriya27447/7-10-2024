import streamlit as st

# กำหนดค่าเริ่มต้นใน session state
if 'foo' not in st.session_state:
    st.session_state.foo = "A 2"
if 'bar' not in st.session_state:
    st.session_state.bar = "B"

# สร้างฟังก์ชันสำหรับหน้า 1
def page1():
    st.title("Page 1")
    st.write("ค่าที่เก็บใน session_state.foo: ", st.session_state.foo)

# สร้างฟังก์ชันสำหรับหน้า 2
def page2():
    st.title("Page 2")
    st.write("ค่าที่เก็บใน session_state.bar: ", st.session_state.bar)

# สร้างการนำทางใน sidebar
page = st.sidebar.selectbox("เลือกหน้า:", ["หน้า 1", "หน้า 2"])

# แสดงเนื้อหาตามหน้า
if page == "หน้า 1":
    page1()
else:
    page2()
