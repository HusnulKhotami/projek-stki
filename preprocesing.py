import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

def preprocess_text(text):
    """
    Fungsi untuk melakukan Text Preprocessing Bahasa Indonesia secara berurutan.
    Menyelaraskan ketentuan: Case Folding, Cleaning, Stopwords, dan Stemming.
    """
    if not isinstance(text, str):
        return ""
        
    #Mengubah huruf menjadi kecil semua
    text = text.lower()
    
    # Filterisasi Menghapus angka, simbol, dan tanda baca menggunakan Regex
    text = re.sub(r'[^a-z\s]', '', text)
    
    #Stopwords Removal Menghapus kata umum yang tidak memuat bobot info penting
    text = stopword_remover.remove(text)
    
    #Stemming memotong kata berimbuhan menjadi kata dasar asli
    text = stemmer.stem(text)
    
    return text