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
Đánh giá chất lượng rượu vang bằng **học máy** giúp con người nhận định chính xác về chất lượng sản phẩm làm ra, từ đó hạn chế những rủi ro trong sản xuất và tăng cao hiệu suất.
Web app này hỗ trợ
- Phân tích dữ liệu
- Tiền xử lý dữ liệu bằng PCA và Normalization
- Đánh giá chất lượng từ tập dữ liệu rượu vang do người dùng nhập vào

## Tầm quan trọng
Trong vài thập kỷ qua, ngành công nghiệp rượu vang phát triển với nhiều chủng loại đa dạng. Trên thực tế, văn hóa rượu vang nói chung đang bùng nổ mạnh mẽ.
    """
        )
        ast.shared.components.video_youtube(
            src="https://www.youtube.com/watch?v=j6VgWHxqaLU&feature=emb_title"
        )