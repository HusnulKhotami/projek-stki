import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocesing import preprocess_text

def load_and_index_dataset(csv_path="data/katalogperpus.csv"):
    """
    Memuat dataset katalog buku dan menghitung matriks pembobotan TF-IDF.
    """
    # Membaca data dari direktori csv sesuai dengan struktur folder Anda
    df = pd.read_csv(csv_path)  
    
    # Melakukan Text Preprocessing pada seluruh kolom Sinopsis
    df['Cleaned_Sinopsis'] = df['Sinopsis'].apply(preprocess_text)
    
    # SOLUSI: Mengaktifkan sublinear_tf untuk meredam dominasi kata berulang 
    # dan norm='l2' untuk mengunci normalisasi panjang dokumen secara adil.
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, 
        norm='l2', 
        smooth_idf=True,
        max_df=0.85,  # Hapus kata yang muncul di lebih dari 85% total dokumen
        min_df=1      # Hanya sertakan kata yang muncul minimal di 1 dokumen
    )
    
    # Menghitung matriks TF-IDF berdasarkan korpus sinopsis buku yang bersih
    tfidf_matrix = vectorizer.fit_transform(df['Cleaned_Sinopsis'])
    
    return df, vectorizer, tfidf_matrix