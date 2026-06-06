import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from preprocesing import preprocess_text  # Pastikan nama file preprocessing.py benar

def dapatkan_pencarian_buku(query, df, vectorizer, tfidf_matrix, top_k=5):
    """
    Menerima query, menghitung kecocokan konteks via Cosine Similarity,
    dan mengembalikan hasil perangkingan variatif (Top-5) beserta token kata pencarian.
    """
    # 1. Bersihkan query masukan pengguna menggunakan modul preprocessing
    cleaned_query = preprocess_text(query)
    
    # Jika query kosong setelah dibersihkan, langsung hentikan proses
    if not cleaned_query.strip():
        return [], []
        
    # Ambil daftar kata kunci unik yang dicari untuk modal efek stabilo
    query_words = list(set(cleaned_query.split()))
    # Masukkan juga kata-kata asli sebelum di-stemming agar pencocokan stabilo lebih kaya
    query_words_raw = list(set(re.sub(r'[^a-zA-Z\s]', '', query.lower()).split()))
    semua_kata_kunci = list(set(query_words + query_words_raw))
        
    # 2. Ubah query menjadi bentuk vektor berbasis kamus TF-IDF dari dataset
    query_vector = vectorizer.transform([cleaned_query])
    
    # 3. Hitung skor Cosine Similarity antara vektor query dengan seluruh sinopsis buku
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # 4. Ambil urutan indeks dokumen dari nilai skor tertinggi ke terendah (descending)
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    hasil_pencarian = []
    rank_counter = 1
    
    # 5. Lakukan perulangan untuk mengambil dokumen dengan skor terbaik
    for idx in ranked_indices:
        # Jika kuota Top-K (Top-5) sudah terpenuhi, hentikan pencarian
        if len(hasil_pencarian) >= top_k:
            break
            
        score = float(similarity_scores[idx])
        
        # Hanya ambil dokumen yang benar-benar memiliki keterikatan kata kunci asli
        if score > 0.001:
            hasil_pencarian.append({
                "Ranking": rank_counter,
                "ID_Buku": df['ID_Buku'].iloc[idx],
                "Judul_Buku": df['Judul_Buku'].iloc[idx],
                "Penulis": df['Penulis'].iloc[idx],
                "Genre": df['Genre'].iloc[idx],
                "Sinopsis_Asli": df['Sinopsis'].iloc[idx],
                "Skor_Relevansi": round(score, 4)
            })
            rank_counter += 1
            
    return hasil_pencarian, semua_kata_kunci