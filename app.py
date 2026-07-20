import streamlit as st

st.set_page_config(
    page_title="Dataset Recommendation System",
    page_icon="📊"
)

st.title("📊 Dataset Recommendation System")

st.write("AI Project MVP")

project = st.text_area(
    "Describe your AI project",
    height=180
)

if st.button("Recommend Dataset"):
    st.success("Recommendation feature will be implemented soon.")