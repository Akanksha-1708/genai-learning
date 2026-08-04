from pydantic import BaseModel,Field

class GeneratedProfiles(BaseModel):
    professional_bio:str=Field(description="A short professional biography")
    linkedin_summary:str=Field(description="Professional linkedin summary")
    github_readme:str=Field(description="Github profile README introduction")
    resume_summary:str=Field(description="Professional resume summary")
    portfolio_about:str=Field(description="Portfolio About Me section")
    elevator_pitch:str=Field(description="30-60 second elevator pitch")
