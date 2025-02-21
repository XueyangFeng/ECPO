import json
import random
import uuid

# Define the structure with single or multiple choice options
book_profile_structure = {
    "Behavioral Traits": {
        "Decision-Making Style": {
            "options": ["Interest-based selection", "Random exploration", "Analytical", "Following trends"],
            "unique": True  # only one decision-making style
        },
        "Emotional Preference": {
            "options": ["Likes suspense", "Prefers warmth", "Favors thrillers", "Optimistic and positive"],
            "unique": True
        }
    },
    "Linguistic Traits": {
        "Information Density": {
            "options": ["Detailed description", "Moderate", "Concise"],
            "unique": True
        },
        "Expression Style": {
            "options": ["Humorous", "Emotionally rich", "Formal", "Casual"],
            "unique": True
        },
        "Tone": {
            "options": ["Enthusiastic and positive", "Critical and serious", "Witty", "Straightforward"],
            "unique": True
        }
    }
}

# Function to generate a single persona by random sampling with mutual exclusion checks
def generate_persona(structure):
    persona = {}
    for category, traits in structure.items():
        persona[category] = {}
        for trait, trait_info in traits.items():
            options = trait_info["options"]
            if trait_info["unique"]:
                # Select a single option
                choice = random.choice(options)
            else:
                # Select multiple options (1 to 3) if non-unique and join them as a single comma-separated string
                choices = random.sample(options, k=random.randint(1, min(3, len(options))))
                choice = ", ".join(choices)

            persona[category][trait] = choice.strip(", ")

    return persona

# Generate 100 personas
personas = []
for _ in range(100):
    persona_data = {
        "UserID": str(uuid.uuid4()),  # unique user ID
        "Persona": generate_persona(book_profile_structure)
    }
    personas.append(persona_data)

# Write personas to JSONL file
output_path = "random_personas.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for persona in personas:
        f.write(json.dumps(persona, ensure_ascii=False) + "\n")

print(f"Generated 100 personas saved to {output_path}")
