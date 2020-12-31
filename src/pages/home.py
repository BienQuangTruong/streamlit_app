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
[Streamlit](https://streamlit.io/) is [announced](https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace) as being **The fastest way to build custom Machine Learning tools** but I believe it has the potential to become much more awesome than that.
I believe Streamlit has the **potential to become the Iphone of Data Science Apps**. And maybe it can even become the Iphone of Technical Writing, Code, Micro Apps and Python.
The **purpose** of the **Awesome Streamlit Project** is to share knowledge on how Awesome Streamlit is and can become.
This application provides
- A list of awesome Streamlit **resources**.
- A **gallery** of awesome streamlit applications.
- A **vision** on how awesome Streamlit can be.
## The Magic of Streamlit
The only way to truly understand how magical Streamlit is to play around with it.
But if you need to be convinced first, then here is the **4 minute introduction** to Streamlit!
Afterwards you can explore examples in the Gallery and go to the [Streamlit docs](https://streamlit.io/docs/) to get started.

Đánh giá chất lượng rượu vang bằng **học máy** giúp con người nhận định chính xác về chất lượng sản phẩm làm ra, từ đó hạn chế những rủi ro trong sản xuất và tăng cao hiệu suất.
Web app này hỗ trợ
- Phân tích dữ liệu
- Tiền xử lý dữ liệu bằng PCA và Normalization
- Đánh giá chất lượng từ tập dữ liệu rượu vang do người dùng nhập vào

## Tầm quan trọng
Trong vài thập kỷ qua, ngành công nghiệp rượu vang phát triển với nhiều chủng loại đa dạng. Trên thực tế, văn hóa rượu vang nói chung đang bùng nổ mạnh mẽ
    """
        )
        ast.shared.components.video_youtube(
            src="https://www.youtube.com/watch?v=j6VgWHxqaLU&feature=emb_title"
        )