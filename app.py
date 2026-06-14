import streamlit as st
import re
import pandas as pd
from indexing import load_and_index_dataset
from retrieval import dapatkan_pencarian_buku
from evaluasi import evaluasi_performa_sistem  


st.set_page_config(
    page_title="Katalog Buku", 
    layout="wide", 
)

@st.cache_resource
def inisialisasi_stki():
    return load_and_index_dataset()

df, vectorizer, tfidf_matrix = inisialisasi_stki()

def beri_efek_stabilo(teks, kata_kunci):
    if not kata_kunci:
        return teks
        
    for kata in kata_kunci:
        if len(kata) > 2:
            pola = re.compile(r'(' + re.escape(kata) + r')', re.IGNORECASE)
            teks = pola.sub(r'<mark style="background-color: #FDE047; color: #1E293B; font-weight: 500; padding: 2px 4px; border-radius: 4px;">\1</mark>', teks)
    return teks


with st.container():
    st.title("Sistem Temu Kembali Informasi")
    st.subheader("Katalog Buku Pintar — Jurusan Ilmu Komputer Universitas Lampung")
    st.write(
        "Sistem pencarian referensi buku berbasis pemodelan *Vector Space Model* (VSM), "
        "pembobotan kata *TF-IDF Sublinear*, dan pengukuran kedekatan sudut dokumen *Cosine Similarity*."
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Membuat Tab Menu navigasi utama tepat bersatu di dalam struktur Header
    tab_home, tab_browse, tab_eval = st.tabs([
        "Beranda & Arsitektur", 
        "Jelajah & Simulator Teks", 
        "Evaluasi Performa MAP"
    ])
    
    # Menghubungkan fungsionalitas Material Icons modern pada Tab bawaan Python
    tab_home.icon = "home"
    tab_browse.icon = "explore"
    tab_eval.icon = "analytics"

st.markdown("---")


# =========================================================
# --- TAB 1: KORPUS DATA & RUANGAN TEORI ---
# =========================================================
with tab_home:
    st.markdown("### Landasan Teori & Korpus Data")
    
    col_edu1, col_edu2 = st.columns(2)
    
    with col_edu1:
        with st.container(border=True):
            st.markdown("**Pipeline Pemrosesan Teks (Model IR)**")
            st.write(
                "Sistem mengekstraksi data tekstual mentah menjadi bentuk vektor ruang "
                "melalui tahapan pipa pemrosesan dokumen terstruktur berikut:"
            )
            st.markdown(
                "- **Tahap Preprocessing:** Pembersihan teks melalui penyeragaman huruf kecil (*Case Folding*), "
                "penghapusan angka/simbol, pemotongan kata hubung (*Stopword Removal*), dan pencarian kata dasar menggunakan Sastrawi (*Stemming*).\n"
                "- **Pembobotan TF-IDF:** Pembobotan kata dengan peredaman frekuensi logaritmik (*Sublinear TF*) untuk menjaga keadilan bobot kata.\n"
                "- **Kalkulasi Kedekatan Vektor:** Menggunakan rumus sudut *Cosine Similarity* antara vektor query dan sinopsis dokumen."
            )
            
    with col_edu2:
        with st.container(border=True):
            st.markdown("**Spesifikasi Dataset Katalog**")
            st.write("Informasi metadata dari data katalog buku pintar Universitas Lampung:")
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label="Nama Database", value="katalogperpus.csv")
            col_m2.metric(label="Jumlah Dokumen", value=f"{len(df)} Buku")
            
            st.markdown("**Schema Atribut Kolom:**")
            st.code("ID_Buku | Judul_Buku | Penulis | Genre | Sinopsis (Korpus Utama)", language="text")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Formulasi Matematika Sistem Temu Kembali Informasi**")
        
        col_math1, col_math2 = st.columns(2)
        with col_math1:
            st.markdown("*1. Pembobotan Term TF-IDF (Sublinear Fitur)*")
            st.latex(r"wf_{t,d} = 1 + \log(\text{tf}_{t,d})")
            
            st.markdown("*2. Perhitungan Kedekatan (Cosine Similarity)*")
            st.latex(r"\text{Cosine Similarity}(Q, D) = \frac{Q \cdot D}{\|Q\| \|D\|}")
            
        with col_math2:
            st.markdown("*3. Kuantitas Hasil Presisi (Precision@K)*")
            st.latex(r"\text{Precision@K} = \frac{|\{\text{Hasil Sistem}\} \cap \{\text{Ground Truth}\}|}{K}")
            
            st.markdown("*4. Kualitas Urutan Ranking (Average Precision)*")
            st.latex(r"\text{AP} = \frac{1}{\min(R, K)} \sum_{k=1}^{K} (\text{Precision}@k \times \text{Rel}(k))")


# =========================================================
# --- TAB 2: JELAJAH DATA, SIMULATOR, & LIVE SEARCH ---
# =========================================================
with tab_browse:
    st.markdown("### Eksplorasi Data dan Simulasi Sistem")
    st.write("Gunakan fitur kotak pencarian terpusat atau filter data di bawah untuk melakukan pengamatan sinopsis buku.")
    
    # Kotak Fitur Search tetap dipertahaman fungsionalitasnya di bagian panel pencarian
    with st.container(border=True):
        query_header = st.text_input(
            "Cari Buku:",
            placeholder="Ketik kata kunci atau topik sinopsis di sini... (Contoh: machine learning, matematika diskrit, kisah cinta)"
        )
        
        if query_header:
            with st.spinner("Sistem sedang menghitung skor kedekatan dokumen..."):
                hasil_header, kata_kunci_header = dapatkan_pencarian_buku(query_header, df, vectorizer, tfidf_matrix, top_k=5)
                if hasil_header:
                    st.success(f"Berhasil menemukan {len(hasil_header)} buku yang relevan dengan query Anda.")
                    for buku in hasil_header:
                        sinopsis_stabilo = beri_efek_stabilo(buku['Sinopsis_Asli'], kata_kunci_header)
                        with st.expander(f"Peringkat {buku['Ranking']} | {buku['Judul_Buku']} (Skor Relevansi: {buku['Skor_Relevansi']})", expanded=True):
                            st.markdown(f"**ID Buku:** `{buku['ID_Buku']}` | **Penulis:** *{buku['Penulis']}* | **Genre:** `{buku['Genre']}`")
                            st.markdown(f"**Sinopsis:** {sinopsis_stabilo}", unsafe_allow_html=True)
                else:
                    st.warning("Tidak ditemukan dokumen sinopsis yang cocok. Silakan gunakan variasi kata kunci lain.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Membagi visualisasi data pasif dan simulator menjadi 2 kolom horizontal
    col_left, col_right = st.columns([1, 1])

    # --- KOLOM KIRI: JELAJAH DATA & KATEGORI ---
    with col_left:
        with st.container(border=True):
            st.markdown("#### Dataset")
            st.write("Saring katalog buku berdasarkan kategori rumpun keilmuan:")
            
            daftar_genre = list(df['Genre'].unique())
            genre_pilihan = st.selectbox("Pilih Genre Buku:", ["Semua Kategori"] + daftar_genre, key="browse_genre_select")
            
            if genre_pilihan == "Semua Kategori":
                df_filtered = df[['ID_Buku', 'Judul_Buku', 'Penulis', 'Genre', 'Sinopsis']]
            else:
                df_filtered = df[df['Genre'] == genre_pilihan][['ID_Buku', 'Judul_Buku', 'Penulis', 'Genre', 'Sinopsis']]
                
            st.dataframe(df_filtered, use_container_width=True, hide_index=True, height=240)
            st.caption(f"Menampilkan {len(df_filtered)} dokumen aktif.")
            
            st.markdown("---")
            st.markdown("#### Temukan Buku Acak")
            st.write("Ambil sampel dokumen acak dari database untuk melihat contoh penulisan korpus:")
            
            if st.button("Rekomendasi Acak Buku Hari Ini", type="secondary", use_container_width=True):
                sampel_acak = df.sample(3)
                for indeks, baris in sampel_acak.iterrows():
                    with st.container(border=True):
                        st.markdown(f"##### 📕 {baris['Judul_Buku']}")
                        st.markdown(f"**ID Buku:** `{baris['ID_Buku']}` | **Genre:** `{baris['Genre']}`")
                        st.markdown(f"**Sinopsis:** {baris['Sinopsis']}")

    # --- KOLOM KANAN: SIMULATOR PREPROCESSING TEKS ---
    with col_right:
        with st.container(border=True):
            st.markdown("#### Simulasi Pipeline Preprocessing Teks")
            st.write("Ketik kalimat kustom untuk mengamati proses sterilisasi kata oleh fungsi `preprocessing.py`:")
            
            kalimat_uji = st.text_input(
                "Masukkan kalimat uji coba bebas:",
                key="simulator_text_input",
                placeholder="Contoh: Mahasiswa sedang melakukan pengujian riset algoritma matematika terapan..."
            )
            
            if kalimat_uji.strip():
                with st.container(border=True):
                    st.markdown("**Analisis Perubahan Struktur Teks Berurutan:**")
                    
                    st.markdown("- **Teks Awal Mentah:**")
                    st.code(kalimat_uji, language="text")
                    
                    teks_cf = kalimat_uji.lower()
                    st.markdown("- **Hasil Tahap 1 (Case Folding):**")
                    st.code(teks_cf, language="text")
                    
                    teks_clean = re.sub(r'[^a-z\s]', '', teks_cf)
                    st.markdown("- **Hasil Tahap 2 (Filtering / Cleaning Simbol):**")
                    st.code(teks_clean, language="text")
                    
                    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
                    factory_sw = StopWordRemoverFactory()
                    remover_sw = factory_sw.create_stop_word_remover()
                    teks_sw = remover_sw.remove(teks_clean)
                    st.markdown("- **Hasil Tahap 3 (Stopword Removal):**")
                    st.code(teks_sw, language="text")
                    
                    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                    factory_st = StemmerFactory()
                    stemmer_st = factory_st.create_stemmer()
                    teks_akhir = stemmer_st.stem(teks_sw)
                    st.markdown("- **Hasil Akhir Tahap 4 (Stemming Kata Dasar):**")
                    st.code(teks_akhir, language="text")
                    
                    st.success("Token berhasil disterilkan dan siap dipetakan ke matriks TF-IDF!")
            else:
                st.info("ℹ️ Silakan ketik kalimat pada kolom input di atas untuk memulai simulasi data interaktif.")


# =========================================================
# --- TAB 3: PANEL PENGUJIAN AKURASI FORMAL ---
# =========================================================
with tab_eval:
    st.markdown("### Pengujian Akurasi Sistem")
    st.write(
        "Bagian ini menguji keandalan sistem menggunakan 10 query dengan konteks berbeda "
        "berdasarkan acuan Ground Truth manual untuk menghitung nilai Precision@5 dan Mean Average Precision (MAP)."
    )
    
    # --- VISUALISASI ALUR KERJA EVALUASI SISTEM (EDUKASI TAMBAHAN) ---
    with st.container(border=True):
        st.markdown("####  Alur Kerja Perhitungan Evaluasi Sistem")
        st.write(
            "Berikut adalah visualisasi pipa pemrosesan (*pipeline workflow*) bagaimana nilai akurasi MAP "
            "dihitung secara otomatis di balik layar sistem:"
        )
        
        # Grid horizontal visualisasi alur
        col_flow1, col_flow2, col_flow3, col_flow4, col_flow5 = st.columns(5)
        
        with col_flow1:
            st.markdown("##### **1. Pemanggilan Query**")
            st.caption("Sistem membaca 10 skenario query pengujian yang telah disiapkan secara seimbang.")
            
        with col_flow2:
            st.markdown("##### **2. Eksekusi VSM**")
            st.caption("Masing-masing query dicari menggunakan modul TF-IDF + Cosine Similarity untuk mengambil Top-5 Rank.")
            
        with col_flow3:
            st.markdown("##### **3. Cek Ground Truth**")
            st.caption("Hasil ID Buku dari sistem dicocokkan dengan list kunci jawaban ideal (Ground Truth) secara biner.")
            
        with col_flow4:
            st.markdown("##### **4. Kalkulasi Metrik**")
            st.caption("Menghitung nilai Precision@5 dan Average Precision (AP) berdasarkan letak urutan rangking dokumen yang benar.")
            
        with col_flow5:
            st.markdown("##### **5. Nilai Akhir MAP**")
            st.caption("Seluruh nilai AP dari ke-10 skenario query dirata-ratakan untuk menghasilkan satu skor akhir MAP global.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Jalankan Pengujian Otomatis", type="primary"):
        with st.spinner("Sedang memproses evaluasi akurasi sistem..."):
            df_ringkasan, skor_map = evaluasi_performa_sistem(df, vectorizer, tfidf_matrix)
            
            col_metric, col_info = st.columns([1, 2])
            with col_metric:
                st.metric(label="Rata-rata Akurasi Sistem (MAP)", value=f"{skor_map}%")
                
            with col_info:
                st.info(
                    f"Pengujian otomatis selesai dilakukan. Nilai Mean Average Precision (MAP) sistem Anda "
                    f"adalah {skor_map}%, dihitung berdasarkan nilai rata-rata dari seluruh skor Average Precision "
                    f"pada 10 skenario query pengujian resmi."
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Tabel Rincian Hasil Pengujian Skenario:**")
            st.dataframe(df_ringkasan, use_container_width=True)