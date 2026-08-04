from pydantic import BaseModel,Field

class UserProfile(BaseModel):
    name:str=Field(...,description="Full name of user")
    education:str=Field(...,description="Educational background")
    skills:list[str]=Field(...,description="List of technical and soft skills")
    projects:list[str]=Field(...,description="Projects completed by user")
    experience:str=Field(...,description="Projects completed by the user")
    achievements:list[str]=Field(...,description="Awards, certifications, hackathons and achievements")
    career_goal:str=Field(...,description="Future career objctive")
    interests:list[str]=Field(...,description="Personal or technical interests")




# (...) in field means : required