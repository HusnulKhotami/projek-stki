
# 📚 Sistem Temu Kembali Informasi Buku Berbasis TF-IDF

Sistem Temu Kembali Informasi (STKI) berbasis web yang dikembangkan untuk membantu pengguna menemukan buku yang relevan berdasarkan isi sinopsis. Sistem menerapkan metode **TF-IDF (Term Frequency–Inverse Document Frequency)** untuk pembobotan kata dan **Cosine Similarity** untuk menghitung tingkat kemiripan antara query pengguna dan dokumen buku.

## 👨‍💻 Tim Pengembang

**Proyek Akhir Sistem Temu Kembali Informasi**

* Azmii Haniifah (2317051028)
* Fitria Nuraini (2317051021)
* Indah Febriana Della (2317051066)
* Husnul Khotami (2317051030)
* Napis Risqullah (2317051019)

## 🎯 Tujuan

* Membantu pengguna menemukan buku tanpa harus mengetahui judul buku secara tepat.
* Melakukan pencarian berdasarkan topik, tema, atau deskripsi buku.
* Menampilkan hasil pencarian berdasarkan tingkat relevansi dokumen.
* Mengukur performa sistem menggunakan metrik evaluasi Information Retrieval.


## 📂 Dataset

Dataset yang digunakan terdiri dari **115 data buku** dengan atribut:

* ID Buku
* Judul Buku
* Penulis
* Genre
* Sinopsis
* Tahun Terbit
* Jumlah Halaman

## ⚙️ Metode yang Digunakan

### 1. Text Preprocessing

* Case Folding
* Cleaning
* Tokenization
* Stopword Removal (Sastrawi)
* Stemming (Sastrawi)

### 2. Pembentukan Corpus

Kumpulan sinopsis buku yang telah melalui preprocessing digunakan sebagai corpus dokumen.

### 3. TF-IDF

Mengubah dokumen teks menjadi representasi numerik berdasarkan bobot kata.

### 4. Cosine Similarity

Mengukur tingkat kemiripan antara query dan dokumen untuk menghasilkan ranking relevansi.

---

## 🚀 Instalasi

Install seluruh dependensi:

```bash
pip install -r requirement.txt
```


## ▶️ Menjalankan Sistem

```bash
streamlit run app.py
```

Aplikasi akan berjalan pada:

```text
http://localhost:8501
```

## 🔍 Alur Sistem

```text
Query Pengguna
      ↓
Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Ranking Dokumen
      ↓
Top-5 Hasil Pencarian
```

## 📊 Evaluasi Sistem

Sistem dievaluasi menggunakan:

* Precision@5
* Average Precision (AP)
* Mean Average Precision (MAP)
* NDCG@5


## 💻 Teknologi

* Python
* Pandas
* NumPy
* Scikit-Learn
* Sastrawi
* Streamlit



Program Studi Ilmu Komputer
Fakultas MIPA
Universitas Lampung

---

⭐ Sistem ini dikembangkan sebagai proyek akhir mata kuliah **Sistem Temu Kembali Informasi (STKI)** dengan implementasi metode **TF-IDF** dan **Cosine Similarity** untuk pencarian buku berdasarkan relevansi isi sinopsis.
