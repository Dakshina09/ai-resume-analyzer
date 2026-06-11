# AI Career Intelligence System

A high-end AI-powered resume analyzer that compares a candidate's resume with a job description and generates ATS scoring, skill-gap analysis, role-fit recommendations, and AI career feedback.

This project is designed as a portfolio-ready machine learning and generative AI application using Streamlit, NLP, semantic similarity, and Gemini AI.

## Features

- Upload resumes in PDF, DOCX, or TXT format
- Compare resume content with a pasted job description
- Generate an ATS-style resume score
- Detect matched and missing skills
- Calculate semantic similarity between resume and job description
- Recommend best-fit career roles
- Evaluate resume structure, project strength, and achievement impact
- Generate AI-powered career feedback using Gemini
- Provide resume improvement suggestions
- Provide a 30-day learning roadmap

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- python-docx
- Sentence Transformers
- scikit-learn
- pandas
- Google Gemini API

## Project Structure

```text
ai-resume-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

## Score Breakdown Explanation

The app calculates an ATS-style score using five components:

| Component | Meaning |
|---|---|
| Skills | Measures how many relevant skills from the job description are found in the resume. |
| Semantic | Measures overall meaning similarity between the resume and job description using sentence embeddings. |
| Impact | Checks whether the resume includes measurable achievements such as percentages, accuracy, revenue, time saved, or improvements. |
| Projects | Checks the strength of project-related content such as built, developed, deployed, model, API, GitHub, and real-world project signals. |
| Structure | Checks whether the resume has important sections like Education, Experience, Skills, and Projects. |

The final ATS score is calculated using this weighted formula:

| Component | Weight |
|---|---:|
| Skill Match | 40% |
| Semantic Similarity | 25% |
| Achievement Impact | 15% |
| Project Strength | 10% |
| Resume Structure | 10% |

Example:

```text
Skills: 0
Semantic: 40
Impact: 29
Projects: 71
Structure: 50
