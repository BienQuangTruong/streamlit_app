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
# Mô hình
<blockquote class="imgur-embed-pub" lang="en" data-id="VZmmxC8"><a href="https://imgur.com/VZmmxC8">View post on imgur.com</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
    """,
        unsafe_allow_html=True,
        )