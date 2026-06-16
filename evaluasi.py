import pandas as pd
from retrieval import dapatkan_pencarian_buku

GROUND_TRUTH_DATA = {
    "komputer meniru berpikir manusia": ["BK-004", "BK-024", "BK-044", "BK-064", "BK-084", "BK-104"],
    
    "algoritma matematika diskrit teori graf": ["BK-007", "BK-027", "BK-047", "BK-067", "BK-087", "BK-107"],
    
    "novel romantis kisah cinta sejati": ["BK-001", "BK-002", "BK-003", "BK-021", "BK-022", "BK-023"],
    
    "petualangan sihir dunia alternatif penguasa kegelapan": ["BK-010", "BK-011", "BK-012", "BK-030", "BK-031"],
    
    "analisis statistik riset skripsi matematika terapan": ["BK-009", "BK-029", "BK-049", "BK-069", "BK-089"],
    
    "cerita rakyat jawa barat janda kaya": ["BK-016", "BK-036", "BK-056", "BK-076", "BK-096"],
    
    "pengembangan aplikasi web framework javascript": ["BK-007", "BK-027", "BK-047", "BK-067", "BK-087"],
    
    "komedi satir revisi skripsi mahasiswa kosan": ["BK-019", "BK-029", "BK-039", "BK-059", "BK-079", "BK-099"],
    
    "machine learning klasifikasi naive bayes gerbang kampus": ["BK-004", "BK-024", "BK-044", "BK-064", "BK-084"],
    
    "desain antarmuka pengguna ui ux figma": ["BK-045", "BK-105", "BK-010", "BK-030", "BK-050"] 
}

def hitung_average_precision(id_buku_sistem, ground_truth):
    """
    Menghitung nilai Average Precision (AP) aktual untuk satu buah query.
    """
    if not id_buku_sistem:
        return 0.0
    
    skor_ap = 0.0
    jumlah_relevan_ditemukan = 0
    
    for k, id_buku in enumerate(id_buku_sistem):
        if id_buku in ground_truth:
            jumlah_relevan_ditemukan += 1
            # Presisi pada posisi ke-k (k diawali dari indeks 0, maka ditambah 1)
            presisi_pada_k = jumlah_relevan_ditemukan / (k + 1)
            skor_ap += presisi_pada_k
            
    # Normalisasi AP dibagi dengan total dokumen relevan yang ada di ground truth (atau min dari sistem)
    total_relevan_ideal = len(ground_truth)
    if jumlah_relevan_ditemukan == 0:
        return 0.0
        
    return skor_ap / min(total_relevan_ideal, len(id_buku_sistem))

def evaluasi_performa_sistem(df, vectorizer, tfidf_matrix):
    """
    Melakukan testing otomatis terhadap 10 query uji untuk menghitung skor Precision@5 dan MAP aktual.
    """
    rekam_evaluasi = []
    total_ap = 0.0
    
    for query, ground_truth in GROUND_TRUTH_DATA.items():
        hasil_sistem, _ = dapatkan_pencarian_buku(query, df, vectorizer, tfidf_matrix, top_k=5)
        id_buku_sistem = [buku['ID_Buku'] for buku in hasil_sistem]
        
        # Menghitung Precision@5 
        relevan_ditemukan = len(set(id_buku_sistem).intersection(set(ground_truth)))
        skor_precision = relevan_ditemukan / len(id_buku_sistem) if len(id_buku_sistem) > 0 else 0.0
        
        # Menghitung Average Precision (AP) yang benar untuk rumusan MAP
        skor_ap = hitung_average_precision(id_buku_sistem, ground_truth)
        total_ap += skor_ap
        
        rekam_evaluasi.append({
            "Query Pengujian": query,
            "Ground Truth": ", ".join(ground_truth),
            "Hasil Top-Sistem": ", ".join(id_buku_sistem) if id_buku_sistem else "Tidak ditemukan",
            "Precision@5": f"{skor_precision * 100:.1f}%",
            "Average Precision (AP)": f"{skor_ap * 100:.1f}%"
        })
        
    # menghitung nilai Mean Average Precision (MAP) secara rata-rata dari seluruh query
    skor_map = (total_ap / len(GROUND_TRUTH_DATA)) * 100
    df_hasil_evaluasi = pd.DataFrame(rekam_evaluasi)
    
    return df_hasil_evaluasi, round(skor_map, 2)