import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python enable_skill.py <skill_name>")
    sys.exit(1)

skill_to_enable = sys.argv[1]
skills_dir = Path(".gemini/skills")
settings_file = Path(".gemini/settings.json")

# Load settings
with open(settings_file, "r") as f:
    settings = json.load(f)

# Get all skills
all_skills = [d.name for d in skills_dir.iterdir() if d.is_dir()]

# Set disabled skills = all skills except the one to enable
settings.setdefault("skills", {})
settings["skills"]["disabled"] = [s for s in all_skills if s != skill_to_enable]

# Save back
with open(settings_file, "w") as f:
    json.dump(settings, f, indent=2)

print(f"Enabled skill: {skill_to_enable}")
print(f"Disabled skills: {settings['skills']['disabled']}")
