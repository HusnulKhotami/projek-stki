# Proyek Akhir STKI - Katalog Buku Pintar Unila

Sistem Temu Kembali Informasi (STKI) berbasis Web Dashboard untuk melakukan pencarian buku pintar berdasarkan relevansi konteks makna pada teks sinopsis menggunakan kombinasi pembobotan **TF-IDF** dan pengukuran sudut **Cosine Similarity**.

## Struktur Repositori
* `data/` : Menyimpan database file Katalog_Buku_STKI_115.csv.
* `preprocessing.py` : Pipa pembersihan teks bahasa Indonesia (Sastrawi).
* `indexing.py` : Pembentukan Vektor Ruang dan hitung matriks TF-IDF.
* `retrieval.py` : Logika pencarian kedekatan sudut dan penentuan ranking.
* `evaluation.py` : Pengujian otomatis performa 10 query (Precision@5 & MAP).
* `app.py` : Antarmuka sistem interaktif berbasis Streamlit Web.

## Cara Menjalankan Aplikasi di Komputer Lokal:
1. Pastikan komputer sudah terinstal Python 3.9 ke atas.
2. Buka terminal/command prompt lalu instal seluruh pustaka dependensi:
   ```bash
   pip install -r requirement.txt

## Cara menjalankan sistem:
streamlit run app.py