"""Model page shown when the user enters the application"""
import streamlit as st

import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Model ..."):
        ast.shared.components.title_awesome("")
        st.write(
            """
<h1 style="color: red">Mô hình</h1>
Trong quá trình xây dựng mô hình, mục đích
của tiền xử lý dữ liệu là để loại bỏ các dữ liệu gây nhiễu. Sau đó, các mẫu phải được phân
nhóm thành bộ hiệu chuẩn và xác nhận dựa trên tiêu chí đã được trình bày trong cả hai bộ.
<br/><br/>
<img src="https://i.imgur.com/VZmmxC8.png" style="max-width: 700px">
    """,
        unsafe_allow_html=True,
        )