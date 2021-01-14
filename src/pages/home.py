"""Home page shown when the user enters the application"""
import streamlit as st

import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Home ..."):
        ast.shared.components.title_awesome("")
        st.write(
            """
# Giới thiệu chung
Đánh giá chất lượng rượu vang bằng **học máy** giúp con người nhận định chính xác về chất lượng sản phẩm làm ra, từ đó hạn chế những rủi ro trong sản xuất và tăng cao hiệu suất.
Web app này hỗ trợ **Đánh giá chất lượng từ tập dữ liệu rượu vang do người dùng nhập vào** theo thang chất lượng: 
- 0 - Không ngon
- 1 - Ngon

## Tầm quan trọng
Trong vài thập kỷ qua, ngành công nghiệp rượu vang phát triển với nhiều chủng loại đa dạng.
    """
        )
        ast.shared.components.video_youtube(
            src="https://www.youtube.com/embed/j6VgWHxqaLU"
            # https://www.youtube.com/embed/B2iAodr0fOo
        )