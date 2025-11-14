"""
LaMini Document Analyzer
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from typing import List, Tuple

# Download once in build (Render runs this)
nltk.download('punkt', quiet=True)

# Fix: Import sent_tokenize
from nltk.tokenize import sent_tokenize

MODEL_NAME = "sentence-transformers/LaMini-LM-L6-v2"

class LaMiniAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = None
        self.chunks = None

    def chunk_text(self, text: str, max_chunk_sentences: int = 6, overlap_sentences: int = 1) -> List[str]:
        sents = sent_tokenize(text)
        if not sents:
            return []
        chunks = []
        i = 0
        while i < len(sents):
            chunk_sents = sents[i:i + max_chunk_sentences]
            chunks.append(" ".join(chunk_sents).strip())
            i += max_chunk_sentences - overlap_sentences
        return chunks

    def embed_chunks(self, chunks: List[str], batch_size: int = 32) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        embs = self.model.encode(chunks, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        self.embeddings = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        self.chunks = chunks
        return self.embeddings

    def fit_document(self, text: str, max_chunk_sentences: int = 6, overlap_sentences: int = 1):
        chunks = self.chunk_text(text, max_chunk_sentences, overlap_sentences)
        embs = self.embed_chunks(chunks)
        return chunks, embs

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[float, str]]:
        if self.embeddings is None:
            raise RuntimeError("Call fit_document first.")
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        scores = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self.chunks[i]) for i in top_idx]

    def extract_key_clauses(self, text: str, keywords: List[str] = None, top_k: int = 5) -> List[Tuple[float, str]]:
        if keywords is None:
            keywords = [
                "contract", "agreement", "breach", "delivery", "notice", "force majeure",
                "वाद", "अनुबंध", "डिलीवरी", "नोटिस", "बाधा", "उल्लंघन", "मुकदमा"
            ]
        chunks = self.chunk_text(text)
        embs = self.embed_chunks(chunks)

        # Keyword score
        kw_scores = []
        lowered = [c.lower() for c in chunks]
        for c in lowered:
            score = sum(1.0 for kw in keywords if kw.lower() in c)
            kw_scores.append(score)

        # Semantic score
        q_emb = self.model.encode(["key clause important legal point"], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        sem_scores = cosine_similarity(q_emb, embs)[0]

        # Normalize & combine
        kw_arr = np.array(kw_scores)
        sem_arr = np.array(sem_scores)
        if kw_arr.max() > 0: kw_arr = kw_arr / kw_arr.max()
        if sem_arr.max() > 0: sem_arr = sem_arr / sem_arr.max()
        combined = 0.4 * kw_arr + 0.6 * sem_arr

        top_idx = np.argsort(-combined)[:top_k]
        return [(float(combined[i]), chunks[i]) for i in top_idx]

# -----------------------
# Example usage (script)
# -----------------------
if __name__ == "__main__":
    sample_hindi = """
    वादी और प्रतिवादी ने 15 जनवरी 2020 को एक आपूर्ति अनुबंध पर हस्ताक्षर किए। 
    पहली खेप 10 फरवरी 2020 को भेजी गई और समय पर पहुंची। 
    25 मार्च 2020 को COVID-19 लॉकडाउन लागू होने के कारण सप्लाई बाधित हो गई। 
    वादी ने 15 अप्रैल 2020 को एक नोटिस भेजा। 
    प्रतिवादी ने 10 मई 2020 को उत्तर दिया कि लॉकडाउन एक 'force majeure' घटना थी। 
    1 जून 2020 को वादी ने अनुबंध उल्लंघन के लिए मुकदमा दायर किया।
    """

    analyzer = LaMiniAnalyzer()
    chunks, embs = analyzer.fit_document(sample_hindi, max_chunk_sentences=4)
    print("Chunks:")
    for i,c in enumerate(chunks): print(i+1, "-", c[:120].replace("\n"," "))

    print("\nTop semantic matches for query 'force majeure':")
    for score,chunk in analyzer.semantic_search("force majeure", top_k=3):
        print(f"{score:.3f} -> {chunk}")

    print("\nExtracted key clauses:")
    for score, clause in analyzer.extract_key_clauses(sample_hindi, top_k=4):
        print(f"{score:.3f} -> {clause[:200].replace('\\n',' ')}")
