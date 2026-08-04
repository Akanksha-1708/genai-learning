from pathlib import Path

from models.input_schema import UserProfile
from chains.profile_generator import generate_profiles

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def clean_list(text: str) -> list[str]:
    """
    Converts comma-separated input into a clean list.
    """
    return [item.strip() for item in text.split(",") if item.strip()]

def get_user_input() -> UserProfile:
    """
    Collect user input and validate it using Pydantic.
    """
    print("=" * 70)
    print("AI PROFILE GENERATOR")
    print("=" * 70)

    return UserProfile(
        name=input("Name: "),
        education=input("Education: "),
        skills=clean_list(input("Skills (comma separated): ")),
        projects=clean_list(input("Projects (comma separated): ")),
        experience=input("Experience: "),
        achievements=clean_list(input("Achievements (comma separated): ")),
        career_goal=input("Career Goal: "),
        interests=clean_list(input("Interests (comma separated): ")),
    )


def main():
    try:
        user = get_user_input()
        profile_data = {
            "name": user.name,
            "education": user.education,
            "skills": ", ".join(user.skills),
            "projects": ", ".join(user.projects),
            "experience": user.experience,
            "achievements": ", ".join(user.achievements),
            "career_goal": user.career_goal,
            "interests": ", ".join(user.interests),
        }

        print("\nGenerating profiles...\n")
        profiles = generate_profiles(profile_data)

        (OUTPUT_DIR / "generated_profiles.json").write_text(
            profiles.model_dump_json(indent=4),
            encoding="utf-8",
        )

        (OUTPUT_DIR / "github_readme.md").write_text(
            profiles.github_readme,
            encoding="utf-8",
        )

        print("=" * 70)
        print("GENERATED PROFILES")
        print("=" * 70)

        print("\n🔹 LinkedIn Summary\n")
        print(profiles.linkedin_summary)

        print("\n🔹 Resume Summary\n")
        print(profiles.resume_summary)

        print("\n🔹 GitHub README\n")
        print(profiles.github_readme)

        print("\n🔹 Portfolio About\n")
        print(profiles.portfolio_about)

        print("\n🔹 Professional Bio\n")
        print(profiles.professional_bio)

        print("\n🔹 Elevator Pitch\n")
        print(profiles.elevator_pitch)

        print("\n" + "=" * 70)
        print("✅ Profiles generated successfully!")
        print("📁 JSON saved to outputs/generated_profiles.json")
        print("📄 GitHub README saved to outputs/github_readme.md")
        print("=" * 70)

    except Exception as e:
        print("\n❌ Error")
        print(e)


if __name__ == "__main__":
    main()