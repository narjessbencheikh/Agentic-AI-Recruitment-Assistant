# rag/indexer.py

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import os



class CVIndexer:
    def __init__(self, vector_store_path: str = "data/vector_store"):
        # Modèle d'embedding - transforme le texte en vecteurs
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # ChromaDB - notre base de données vectorielle locale
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.collection = self.client.get_or_create_collection(
            name="cvs",
            metadata={"hnsw:space": "cosine"}  # similarité cosinus
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrait le texte brut d'un PDF."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """
        Découpe le texte en chunks avec overlap.
        overlap = les 50 derniers mots d'un chunk
        sont répétés au début du suivant
        pour ne pas perdre le contexte entre chunks.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def index_cvs(self, cvs_folder: str = "data/cvs") -> int:
        """
        Indexe tous les CVs PDF du dossier.
        Retourne le nombre de CVs indexés.
        """
        cv_files = list(Path(cvs_folder).glob("*.pdf"))
        
        if not cv_files:
            print("❌ No PDF files found in", cvs_folder)
            return 0
        
        total_indexed = 0
        
        for cv_path in cv_files:
            print(f"📄 Indexing: {cv_path.name}")
            
            # 1. Extraire le texte
            text = self.extract_text_from_pdf(str(cv_path))
            
            if not text:
                print(f"⚠️ Could not extract text from {cv_path.name}")
                continue
            
            # 2. Découper en chunks
            chunks = self.chunk_text(text)
            
            # 3. Générer les embeddings
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # 4. Stocker dans ChromaDB
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                self.collection.add(
                    ids=[f"{cv_path.stem}_chunk_{i}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": cv_path.name, "chunk_index": i}]
                )
            
            total_indexed += 1
            print(f"✅ {cv_path.name} — {len(chunks)} chunks indexed")
        
        print(f"\n✅ Total: {total_indexed} CVs indexed")
        return total_indexed