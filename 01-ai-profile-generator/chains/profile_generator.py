from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from prompts.prompt import (
    linkedin_prompt,
    resume_prompt,
    github_prompt,
    portfolio_prompt,
    bio_prompt,
    pitch_prompt,
)

from models.output_schema import GeneratedProfiles
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

linkedin_chain = linkedin_prompt | model | parser
resume_chain = resume_prompt | model | parser
github_chain = github_prompt | model | parser
portfolio_chain = portfolio_prompt | model | parser
bio_chain = bio_prompt | model | parser
pitch_chain = pitch_prompt | model | parser

profile_generator = RunnableParallel(
    linkedin_summary=linkedin_chain,
    resume_summary=resume_chain,
    github_readme=github_chain,
    portfolio_about=portfolio_chain,
    professional_bio=bio_chain,
    elevator_pitch=pitch_chain,
)

def generate_profiles(user_data: dict) -> GeneratedProfiles:
    """
    Generate all professional profiles using RunnableParallel.
    """
    result = profile_generator.invoke(user_data)
    return GeneratedProfiles(**result)