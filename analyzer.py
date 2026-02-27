from typing import List
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import re
from datetime import datetime

load_dotenv()

llm = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0,
)

# ---------------- Structured Output ---------------- #

class ResumeAnalysis(BaseModel):
    overall_match_score: int = Field(description="Score from 0-100")
    required_skills: List[str]
    candidate_skills: List[str]
    matching_skills: List[str]
    missing_skills: List[str]
    improvement_suggestions: List[str]


# ---------------- Experience Extraction ---------------- #

def extract_experience_range(job_text: str):
    range_pattern = r"(\d+)\s*[-to]+\s*(\d+)\s*years"
    plus_pattern = r"(\d+)\+?\s*years"

    job_text = job_text.lower()

    range_match = re.search(range_pattern, job_text)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    plus_match = re.search(plus_pattern, job_text)
    if plus_match:
        val = int(plus_match.group(1))
        return val, val

    return 0, 0


def extract_candidate_experience(resume_text: str):
    """
    Extract experience based on date ranges like:
    Jul 2024 – Dec 2025
    Feb 2024 - Jun 2024
    """

    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }

    pattern = r"([A-Za-z]{3})\s*(\d{4})\s*[–-]\s*([A-Za-z]{3})\s*(\d{4})"

    matches = re.findall(pattern, resume_text)

    total_months = 0

    for match in matches:
        start_month = month_map.get(match[0].lower(), 1)
        start_year = int(match[1])
        end_month = month_map.get(match[2].lower(), 1)
        end_year = int(match[3])

        start_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)

        diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        if diff > 0:
            total_months += diff

    # Convert months to years
    total_years = round(total_months / 12, 1)

    return total_years


# ---------------- Main Analyzer ---------------- #

def analyze_resume(resume_text: str, job_description: str):

    # Extract experience deterministically
    required_min, required_max = extract_experience_range(job_description)
    candidate_exp = extract_candidate_experience(resume_text)

    # Deterministic experience gap logic
    if required_min == 0 and required_max == 0:
        gap_analysis = "Experience requirement not clearly specified."
    elif candidate_exp < required_min:
        gap_analysis = "Candidate does not meet the minimum required experience."
    elif required_min <= candidate_exp <= required_max:
        gap_analysis = "Candidate meets the required experience range."
    else:
        gap_analysis = "Candidate exceeds the required experience range."

    # LLM Skill Analysis
    prompt = ChatPromptTemplate.from_template("""
            You are a senior technical recruiter.

            Strictly compare resume and job description.

            Extract:
            - Required skills
            - Candidate skills
            - Matching skills
            - Missing skills
            - Provide realistic match score (strict evaluation)
            - Provide strong improvement suggestions

            Resume:
            {resume}

            Job Description:
            {job}

            Return structured JSON only.
            """)

    structured_llm = llm.with_structured_output(ResumeAnalysis)
    chain = prompt | structured_llm

    result = chain.invoke({
        "resume": resume_text,
        "job": job_description
    })

    return {
        "overall_match_score": result.overall_match_score,
        "required_skills": result.required_skills,
        "candidate_skills": result.candidate_skills,
        "matching_skills": result.matching_skills,
        "missing_skills": result.missing_skills,
        "required_experience_min": required_min,
        "required_experience_max": required_max,
        "candidate_experience_years": candidate_exp,
        "experience_gap_analysis": gap_analysis,
        "improvement_suggestions": result.improvement_suggestions
    }