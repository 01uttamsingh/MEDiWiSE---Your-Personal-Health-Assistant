# bot_generic_health_ai_professional.py

import google.generativeai as genai

# Initialize Gemini with your API key
genai.configure(api_key="YOUR API KEY")

# Initialize the model
model = genai.GenerativeModel("models/gemini-2.5-flash")

def chat_with_health_assistant(user_input: str):
    """
    Takes a user's input and generates a professional and empathetic health-assistant response.
    Avoids repeating the bot's name or company.
    """
    prompt = f"""
    You are a professional and empathetic AI health assistant.
    Respond clearly, politely, and concisely. Avoid mentioning your name or company repeatedly.
    User says: "{user_input}"
    Give helpful information, guidance, or general health suggestions without diagnosing or giving medical advice.
    """
    response = model.generate_content(prompt)
    return response.text.replace('*', '')

# Example usage: just call this function with any message
if __name__ == "__main__":
    print("Hello!! Welcome to MedicoAi. I'm your AI health assistant, Aaru. How may I help you today?")
    message = input()
    answer = chat_with_health_assistant(message)
    print(answer)
