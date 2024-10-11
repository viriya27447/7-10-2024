import streamlit as st
import streamlit.components.v1 as components

# สร้างป๊อปอัปด้วย HTML
html_code = """
<div style="background-color: black; padding: 10px; border-radius: 5px;">
    <h2 style="color: white;">นี่คือป๊อปอัปที่กำหนดเอง!</h2>
    <p style="color: white;">คุณสามารถใช้ HTML, CSS และ JavaScript ที่คุณต้องการได้!</p>
</div>
"""

# แสดง HTML
components.html(html_code, height=200)
