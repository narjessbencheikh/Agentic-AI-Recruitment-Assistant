# rag/generator.py

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from rag.retriever import CVRetriever
from dotenv import load_dotenv
import os
from rag.prompt_templates import CV_GENERATION_PROMPT


load_dotenv()

class CVGenerator:
    def __init__(self):
        self.llm = OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("MODEL_NAME")
        )
        self.retriever = CVRetriever()
        
        self.prompt = PromptTemplate(
            input_variables=["job_description", "candidate_profile", "context", "skills"],
            template=CV_GENERATION_PROMPT
        )
    
    def generate(self, job_description: str, candidate_profile: str, skills: dict) -> str:
        """
        Génère un CV optimisé basé sur le poste et le profil candidat.
        
        Args:
            job_description: Texte de l'offre d'emploi
            candidate_profile: Profil brut du candidat
            skills: Compétences extraites par SkillExtractorAgent
            
        Returns:
            CV optimisé en texte
        """
        # 1. Retrieval — chercher des profils similaires
        context_results = self.retriever.retrieve(job_description, top_k=3)
        context = "\n---\n".join([r["text"] for r in context_results])
        
        # 2. Formatter les skills
        skills_text = f"""
        Must have: {", ".join(skills.get("must_have", []))}
        Nice to have: {", ".join(skills.get("nice_to_have", []))}
        Soft skills: {", ".join(skills.get("soft_skills", []))}
        """
        
        # 3. Générer le CV
        chain = self.prompt | self.llm
        response = chain.invoke({
            "job_description": job_description,
            "candidate_profile": candidate_profile,
            "context": context,
            "skills": skills_text
        })
        
        return response