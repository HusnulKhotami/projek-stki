import pandas as pd
from retrieval import dapatkan_pencarian_buku

# Definisikan 10 skenario query pengujian yang relevan secara kontekstual
GROUND_TRUTH_DATA = {
    "komputer meniru berpikir manusia": ["BK-001", "BK-021", "BK-041", "BK-061", "BK-081", "BK-101"],
    "algoritma matematika diskrit teori graf": ["BK-007", "BK-027", "BK-047", "BK-067", "BK-087", "BK-107"],
    "novel romantis kisah cinta sejati": ["BK-001", "BK-002", "BK-003", "BK-021", "BK-022", "BK-023"],
    "petualangan sihir dunia alternatif penguasa kegelapan": ["BK-010", "BK-011", "BK-012", "BK-030", "BK-031"],
    "analisis statistik riset skripsi matematika terapan": ["BK-009", "BK-029", "BK-049", "BK-069", "BK-089"],
    "cerita rakyat jawa barat janda kaya": ["BK-016", "BK-017", "BK-018", "BK-036", "BK-037"],
    "pengembangan aplikasi web framework javascript": ["BK-007", "BK-027", "BK-047", "BK-067", "BK-087"],
    "komedi satir revisi skripsi mahasiswa kosan": ["BK-019", "BK-020", "BK-039", "BK-040", "BK-059"],
    "machine learning klasifikasi naive bayes gerbang kampus": ["BK-004", "BK-024", "BK-044", "BK-064", "BK-084"],
    "desain antarmuka pengguna ui ux figma": ["BK-010", "BK-030", "BK-050", "BK-070", "PN-090"]
}

def evaluasi_performa_sistem(df, vectorizer, tfidf_matrix):
    """
    Melakukan testing otomatis terhadap 10 query uji untuk menghitung skor Precision@5 dan MAP.
    """
    rekam_evaluasi = []
    total_precision = 0.0
    
    for query, ground_truth in GROUND_TRUTH_DATA.items():
        # SOLUSI: Menambahkan variabel '_' untuk menampung token kata kunci agar list buku 'hasil_sistem' tidak rusak
        hasil_sistem, _ = dapatkan_pencarian_buku(query, df, vectorizer, tfidf_matrix, top_k=5)
        
        # Mengambil ID_Buku dari list hasil sistem yang kini formatnya sudah benar
        id_buku_sistem = [buku['ID_Buku'] for buku in hasil_sistem]
        
        # Hitung irisan kecocokan antara output sistem dengan Ground Truth ideal
        relevan_ditemukan = len(set(id_buku_sistem).intersection(set(ground_truth)))
        
        # Hitung Precision pada Top-5
        skor_precision = relevan_ditemukan / len(id_buku_sistem) if len(id_buku_sistem) > 0 else 0.0
        total_precision += skor_precision
        
        rekam_evaluasi.append({
            "Query Pengujian": query,
            "Ground Truth": ", ".join(ground_truth),
            "Hasil Top-Sistem": ", ".join(id_buku_sistem) if id_buku_sistem else "Tidak ditemukan",
            "Precision@5": f"{skor_precision * 100:.1f}%"
        })
        
    # Hitung nilai Mean Average Precision (MAP) secara rata-rata
    skor_map = (total_precision / len(GROUND_TRUTH_DATA)) * 100
    df_hasil_evaluasi = pd.DataFrame(rekam_evaluasi)
    
    return df_hasil_evaluasi, round(skor_map, 2)