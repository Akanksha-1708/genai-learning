from langchain_core.prompts import PromptTemplate

PROFILE_CONTEXT = """
User Profile
Name:
{name}
Education:
{education}
Skills:
{skills}
Projects:
{projects}
Experience:
{experience}
Achievements:
{achievements}
Career Goal:
{career_goal}
Interests:
{interests}
"""

linkedin_prompt = PromptTemplate.from_template(
f"""
You are an experienced LinkedIn profile optimization expert.

Your task is to write a professional LinkedIn summary using ONLY the information provided.

Rules:
- Do NOT invent companies, certifications, experience, projects or achievements.
- Do NOT add technologies not mentioned.
- Write in FIRST person.
- Professional and engaging tone.
- Highlight strengths naturally.
- Mention future aspirations.
- Keep between 120-180 words.

{PROFILE_CONTEXT}

Return ONLY the LinkedIn summary.
"""
)

resume_prompt = PromptTemplate.from_template(
f"""
You are an ATS resume writing expert.

Write a resume summary.

Rules:
- Do NOT mention the candidate's name.
- Do NOT invent experience.
- Keep under 100 words.
- Professional tone.
- Mention education.
- Mention strongest technical skills.
- Mention projects briefly.
- Mention career goal naturally.
- ATS friendly.

{PROFILE_CONTEXT}

Return ONLY the resume summary.
"""
)

github_prompt = PromptTemplate.from_template(
f"""
You are writing a GitHub Profile README.

Rules:
- Return valid Markdown.
- Friendly tone.
- Do NOT invent repositories.
- Mention only provided skills.
- Mention only provided projects.
- Include these sections:
# Hi 👋 I'm {{name}}
## 🚀 About Me
## 💻 Tech Stack
## 🌱 Currently Learning
## 📌 Projects
## 🎯 Career Goal
## 🤝 Let's Connect
End with a friendly closing sentence.

{PROFILE_CONTEXT}

Return ONLY Markdown.
"""
)

portfolio_prompt = PromptTemplate.from_template(
f"""
You are writing the About section of a software engineer's portfolio.

Rules:
- Friendly yet professional.
- Do NOT copy resume style.
- Sound human.
- Mention passion for technology.
- Mention projects.
- Mention skills.
- Mention future goal.
- Around 150 words.

{PROFILE_CONTEXT}

Return ONLY the About Me section.
"""
)

bio_prompt = PromptTemplate.from_template(
f"""
You are writing a short professional biography.

Rules:
- Third person.
- Around 80-100 words.
- Mention education.
- Mention technical strengths.
- Mention achievements.
- Mention career goal.
- Do NOT invent facts.

{PROFILE_CONTEXT}

Return ONLY the biography.
"""
)

pitch_prompt = PromptTemplate.from_template(
f"""
You are preparing a 30-second elevator pitch.

Rules:
- First person.
- Conversational.
- Sound natural.
- Mention education.
- Mention strongest skills.
- Mention one important project.
- Mention career goal.
- Around 80 words.
- Do NOT sound like reading a resume.

{PROFILE_CONTEXT}

Return ONLY the elevator pitch.
"""
)