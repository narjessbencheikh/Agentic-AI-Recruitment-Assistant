# app.py

import streamlit as st
from agents.orchestrator import OrchestratorAgent
from rag.indexer import CVIndexer
from rag.generator import CVGenerator
from evaluation.metrics import RAGEvaluator

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AI Recruitment Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Recruitment Assistant")
st.markdown("*Powered by Mistral 7B + RAG + Multi-Agent AI*")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of similar profiles to retrieve", 1, 10, 3)
    st.divider()
    
    st.header("📂 Index CVs")
    if st.button("🔄 Index CVs from data/cvs/"):
        with st.spinner("Indexing CVs..."):
            indexer = CVIndexer()
            total = indexer.index_cvs()
            st.success(f"✅ {total} CVs indexed successfully!")

# ============================================
# MAIN TABS
# ============================================
tab1, tab2, tab3 = st.tabs([
    "🔍 Job Analysis", 
    "📄 CV Generator", 
    "📊 Evaluation"
])

# ============================================
# TAB 1 — JOB ANALYSIS
# ============================================
with tab1:
    st.header("🔍 Job Description Analysis")
    
    job_description = st.text_area(
        "Paste the job description here",
        height=300,
        placeholder="We are looking for a Senior Data Scientist..."
    )
    
    if st.button("🚀 Analyze Job", type="primary"):
        if job_description:
            with st.spinner("Agents are working..."):
                orchestrator = OrchestratorAgent()
                result = orchestrator.run(job_description)
            
            if result["error"]:
                st.error(f"❌ {result['error']}")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Job Analysis")
                    job = result["job_analysis"]
                    st.metric("Job Title", job.get("Job title", "N/A"))
                    st.metric("Sector", job.get("Sector/Industry", "N/A"))
                    st.metric("Experience Level", job.get("Experience level required", "N/A"))
                    st.write("**Key Responsibilities:**")
                    for r in job.get("Key responsibilities", []):
                        st.write(f"• {r}")
                
                with col2:
                    st.subheader("🛠️ Skills Extracted")
                    skills = result["skills"]
                    
                    st.write("**Must Have:**")
                    for s in skills.get("must_have", []):
                        st.badge(s)
                    
                    st.write("**Nice to Have:**")
                    for s in skills.get("nice_to_have", []):
                        st.badge(s)
                    
                    st.write("**Soft Skills:**")
                    for s in skills.get("soft_skills", []):
                        st.badge(s)
                
                st.subheader("👤 Benchmark Profile")
                benchmark = result["benchmark_profile"]
                st.info(f"**Ideal Background:** {benchmark.get('ideal_background', 'N/A')}")
                st.write(f"**Typical Experience:** {benchmark.get('typical_experience_years', 'N/A')}")
                
                col3, col4 = st.columns(2)
                with col3:
                    st.write("**Differentiating Factors:**")
                    for f in benchmark.get("differentiating_factors", []):
                        st.write(f"✅ {f}")
                with col4:
                    st.write("**Red Flags:**")
                    for f in benchmark.get("red_flags", []):
                        st.write(f"🚩 {f}")
                
                # Save to session state for CV Generator
                st.session_state["job_description"] = job_description
                st.session_state["skills"] = skills
        else:
            st.warning("Please enter a job description first.")

# ============================================
# TAB 2 — CV GENERATOR
# ============================================
with tab2:
    st.header("📄 CV Generator")
    
    job_input = st.text_area(
        "Job Description",
        value=st.session_state.get("job_description", ""),
        height=200
    )
    
    candidate_profile = st.text_area(
        "Candidate Profile",
        height=200,
        placeholder="I am a data scientist with 3 years of experience in Python..."
    )
    
    if st.button("✨ Generate Optimized CV", type="primary"):
        if job_input and candidate_profile:
            with st.spinner("Generating your optimized CV..."):
                skills = st.session_state.get("skills", {
                    "must_have": [],
                    "nice_to_have": [],
                    "soft_skills": []
                })
                generator = CVGenerator()
                cv = generator.generate(job_input, candidate_profile, skills)
            
            st.success("✅ CV Generated!")
            st.markdown("---")
            st.markdown(cv)
            
            # Download button
            st.download_button(
                label="📥 Download CV",
                data=cv,
                file_name="optimized_cv.txt",
                mime="text/plain"
            )
            
            # Save for evaluation
            st.session_state["generated_cv"] = cv
            st.session_state["job_input"] = job_input
        else:
            st.warning("Please fill in both fields.")

# ============================================
# TAB 3 — EVALUATION
# ============================================
with tab3:
    st.header("📊 Evaluation Metrics")
    
    relevant_sources = st.text_input(
        "Relevant CV filenames (comma separated)",
        placeholder="cv1.pdf, cv2.pdf"
    )
    
    required_skills = st.text_input(
        "Required skills (comma separated)",
        placeholder="Python, TensorFlow, Docker"
    )
    
    if st.button("📈 Run Evaluation", type="primary"):
        generated_cv = st.session_state.get("generated_cv", "")
        job_input = st.session_state.get("job_input", "")
        
        if generated_cv and job_input and relevant_sources and required_skills:
            with st.spinner("Running evaluation..."):
                evaluator = RAGEvaluator()
                report = evaluator.evaluate(
                    query=job_input,
                    relevant_sources=[s.strip() for s in relevant_sources.split(",")],
                    generated_cv=generated_cv,
                    job_description=job_input,
                    required_skills=[s.strip() for s in required_skills.split(",")]
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔍 Retrieval Metrics")
                retrieval = report["retrieval"]
                st.metric("Precision@K", retrieval["precision@k"])
                st.metric("Recall@K", retrieval["recall@k"])
                st.metric("MRR", retrieval["mrr"])
            
            with col2:
                st.subheader("✍️ Generation Metrics")
                generation = report["generation"]
                st.metric("Relevance Score", generation["relevance_score"])
                st.metric("Skill Match Score", generation["skill_match_score"])
        else:
            st.warning("Please generate a CV first in the CV Generator tab.")