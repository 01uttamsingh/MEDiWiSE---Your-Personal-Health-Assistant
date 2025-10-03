# utils/chatbots.py
from chatbots.bot_remedies import get_home_remedies
from chatbots.bot_generic import chat_with_health_assistant

# You can optionally add wrapper functions if you want
def get_remedies(disease):
    return get_home_remedies(disease)

def get_generic_response(message):
    return chat_with_health_assistant(message)
