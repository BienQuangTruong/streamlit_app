"""Main module for the streamlit app"""
import streamlit as st

import awesome_streamlit as ast
import src.pages.about
import src.pages.gallery.index
import src.pages.home
import src.pages.vision
import src.pages.xuLyData
import src.pages.regressor_module
import src.pages.classifier_module
import src.pages.tienXuLyData
import src.pages.modelPage

# ast.core.services.other.set_logging_format()

PAGES = {
    "Home": src.pages.home,
    "Phân tích dữ liệu": src.pages.xuLyData,
    "Tiền xử lý": src.pages.tienXuLyData,
    "Mô hình": src.pages.modelPage,
    "Hồi quy (Regression)": src.pages.regressor_module,
    "Phân lớp (Classifier)": src.pages.classifier_module
}

def main():
    """Main function of the App"""
    st.sidebar.title("MỤC LỤC")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)
    st.sidebar.title("Intro")
    st.sidebar.info(
        "App này được làm ra nhằm mục đích phục vụ cho việc **nghiên cứu và học tập** "
        "[source code](https://github.com/BienQuangTruong/streamlit_app). "
    )
    st.sidebar.title("About")
    st.sidebar.info(
        """
        App này được thực hiện bởi Biện Quang Trường - Nguyễn Văn Phước.
"""
    )

if __name__ == "__main__":
    main()