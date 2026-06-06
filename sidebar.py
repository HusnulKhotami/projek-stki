import streamlit as st

#notasi objek
pilih = st.sidebar.selectbox(
    "Pilih model yang ingin digunakan",
    ("KNN, SVM, CNN")
)

#gunakan with
with st.sidebar:
    pilih_opsi = st.radio(
        "Pilih tipe data",
        ("Numerik", "Kategorik")
    )