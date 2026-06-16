import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocesing import preprocess_text

def load_and_index_dataset(csv_path="data/katalogperpus.csv"):
    """
    Memuat dataset katalog buku dan menghitung matriks pembobotan TF-IDF.
    """
    df = pd.read_csv(csv_path)  
    
    # Melakukan Text Preprocessing pada seluruh kolom Sinopsis
    df['Cleaned_Sinopsis'] = df['Sinopsis'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, 
        norm='l2', 
        smooth_idf=True,
        max_df=0.85,
        min_df=1      
    )
    
    # Menghitung matriks TF-IDF berdasarkan sinopsis buku
    tfidf_matrix = vectorizer.fit_transform(df['Cleaned_Sinopsis'])
    
    return df, vectorizer, tfidf_matrix