
import google.generativeai as genai

# Initialize Gemini with API key directly
genai.configure(api_key="Your API KEY")

# Initialize the model
model = genai.GenerativeModel("models/gemini-2.5-flash")

def get_home_remedies(disease_name: str):
    """
    Given a disease name, this function asks Gemini API to suggest
    3-5 simple home remedies.
    """
    prompt = f"""
    You are a friendly health assistant.
    Suggest 3-5 simple home remedies that a person can try at home for the disease: {disease_name}.
    Keep the instructions clear and easy to follow.
    """

    response = model.generate_content(prompt)

    # Return the text reply without markdown symbols
    return response.text.replace('*', '')

if __name__ == "__main__":
    disease = input("Enter the disease name: ")
    remedies = get_home_remedies(disease)
    print("\nHome Remedies:")
    print(remedies)
