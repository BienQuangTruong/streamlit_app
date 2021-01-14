"""Main module for the streamlit app"""
import streamlit as st
import awesome_streamlit as ast
import src.pages.home
import src.pages.classifier_module

import src.pages.predict

# ast.core.services.other.set_logging_format()

PAGES = {
    "Giới thiệu": src.pages.home,
    "Huấn luyện mô hình": src.pages.classifier_module,
    "Dự đoán": src.pages.predict
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