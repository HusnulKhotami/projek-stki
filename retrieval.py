import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from preprocesing import preprocess_text  # Pastikan nama file preprocessing.py benar

def dapatkan_pencarian_buku(query, df, vectorizer, tfidf_matrix, top_k=5):
   
    cleaned_query = preprocess_text(query)
    
    if not cleaned_query.strip():
        return [], []
        
    # Ambil daftar kata kunci unik yang dicari untuk modal efek stabilo
    query_words = list(set(cleaned_query.split()))
    query_words_raw = list(set(re.sub(r'[^a-zA-Z\s]', '', query.lower()).split()))
    semua_kata_kunci = list(set(query_words + query_words_raw))
        
    #Mengubah query menjadi bentuk vektor berbasis kamus TF-IDF dari dataset
    query_vector = vectorizer.transform([cleaned_query])
    
    #Mwnghitung skor Cosine Similarity antara vektor query dengan seluruh sinopsis buku
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    hasil_pencarian = []
    rank_counter = 1
    
    # Melakukan perulangan untuk mengambil dokumen dengan skor terbaik
    for idx in ranked_indices:
        if len(hasil_pencarian) >= top_k:
            break
            
        score = float(similarity_scores[idx])
        
        # Mengambil dokumen yang benar-benar memiliki keterikatan kata kunci asli
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