import streamlit as st
import re
from indexing import load_and_index_dataset
from retrieval import dapatkan_pencarian_buku
from evaluasi import evaluasi_performa_sistem  

# Seting Konfigurasi Tampilan Halaman Web Dashboard
st.set_page_config(page_title="STKI Katalog Buku Unila", layout="wide", page_icon="")

# Menggunakan fitur caching Streamlit agar data tidak di-reindex terus-menerus setiap halaman di-refresh
@st.cache_resource
def inisialisasi_stki():
    return load_and_index_dataset()

# Memuat data awal, vectorizer, dan matriks matematika TF-IDF
df, vectorizer, tfidf_matrix = inisialisasi_stki()

def beri_efek_stabilo(teks, kata_kunci):
    """
    Fungsi untuk membungkus kata yang cocok dengan query menggunakan tag HTML <mark>
    Tanpa merusak susunan huruf besar/kecil asli di dalam sinopsis.
    """
    if not kata_kunci:
        return teks
        
    for kata in kata_kunci:
        if len(kata) > 2:  # Hindari menstabilo kata pendek di bawah 3 huruf seperti 'di', 'ke'
            # Menggunakan regex kata utuh (\b) atau pencocokan parsial yang aman
            pola = re.compile(r'(' + re.escape(kata) + r')', re.IGNORECASE)
            teks = pola.sub(r'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">\1</mark>', teks)
    return teks

st.title("Sistem Temu Kembali Informasi - Katalog Buku Pintar")
st.markdown("### Jurusan Ilmu Komputer, Universitas Lampung")
st.write("Cari referensi buku akademik maupun fiksi berdasarkan kesamaan konteks makna pada sinopsis.")

# Membuat Menu Berbentuk Tab Navigasi Sederhana
tab_pencarian, tab_evaluasi = st.tabs([" Fitur Mesin Pencari", " Evaluasi Performa Sistem"])

# --- TAB 1: LOGIKA INTERFACE PENCARIAN UTAMA ---
with tab_pencarian:
    st.subheader("Pencarian Berbasis Konteks")
    query_pengguna = st.text_input("Ketikkan kata kunci, topik, atau penggalan deskripsi sinopsis buku:")
    
    if st.button("Mulai Cari Buku"):
        if query_pengguna.strip():
            # Memanggil modul retrieval untuk langsung memunculkan 5 urutan buku terbaik dan daftar kata kuncinya
            hasil, kata_kunci_pencari = dapatkan_pencarian_buku(query_pengguna, df, vectorizer, tfidf_matrix, top_k=5)
            
            if hasil:
                st.success(f"Berhasil menemukan {len(hasil)} buku yang relevan dengan kriteria Anda!")
                
                # Merender hasil dalam bentuk list kartu informasi vertikal
                for buku in hasil:
                    with st.expander(f"🏅 **[RANK {buku['Ranking']}] - {buku['Judul_Buku']}** (Skor Relevansi: {buku['Skor_Relevansi']})", expanded=True):
                        st.markdown(f"**ID Buku:** `{buku['ID_Buku']}` | **Penulis:** *{buku['Penulis']}* | **Genre/Kategori:** `{buku['Genre']}`")
                        
                        # Memproses teks sinopsis asli agar disuntikkan efek stabilo html
                        sinopsis_stabilo = beri_efek_stabilo(buku['Sinopsis_Asli'], kata_kunci_pencari)
                        
                        # Tampilkan sinopsis menggunakan fungsi markdown html yang diizinkan (unsafe_allow_html=True)
                        st.markdown(f"**Sinopsis Buku:** {sinopsis_stabilo}", unsafe_allow_html=True)
            else:
                st.warning("Tidak ditemukan kecocokan buku. Silakan masukkan kata kunci konteks yang lain.")
        else:
            st.error("Silakan masukkan teks pencarian terlebih dahulu!")

# --- TAB 2: LOGIKA INTERFACE PANEL TESTING EVALUASI ---
with tab_evaluasi:
    st.subheader("Modul Pengujian Akurasi Formal")
    st.write("Bagian ini menguji keandalan sistem menggunakan 10 query dengan konteks berbeda berdasarkan acuan Ground Truth manual.")
    
    if st.button("Jalankan Pengujian Otomatis ⚙️"):
        with st.spinner("Sedang menghitung performa akurasi sistem..."):
            df_ringkasan, skor_map = evaluasi_performa_sistem(df, vectorizer, tfidf_matrix)
            
            # Menampilkan Nilai Rata-Rata Akurasi Sistem (MAP)
            st.metric(label="Rata-rata Akurasi Sistem (Mean Average Precision - MAP)", value=f"{skor_map}%")
            
            # Tampilkan Tabel Komparasi Hasil Pengujian Lengkap
            st.write("### Tabel Rincian Hasil Pengujian Skenario:")
            st.dataframe(df_ringkasan, use_container_width=True)