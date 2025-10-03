import os
import io
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

# --- Imports for Deep Learning Model ---
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# --- 1. INITIALIZATION AND CONFIGURATION ---
load_dotenv()
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not configure Gemini API. Details: {e}")


# --- 2. LOAD ALL MODELS AND DATA ---

# Symptom Prediction (ML) Models
try:
    symptom_model = joblib.load(os.path.join(basedir, 'models/symptom_disease_model_rf.pkl'))
    symptom_encoder = joblib.load(os.path.join(basedir, 'models/symptom_encoder.pkl'))
    label_encoder = joblib.load(os.path.join(basedir, 'models/label_encoder.pkl'))
    print("✅ Symptom prediction models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading symptom models: {e}")
    symptom_model = None

# Disease Precautions Data for Symptom Prediction
try:
    precaution_df = pd.read_csv(os.path.join(basedir, "datasets/Disease precaution.csv"))
    print("✅ Symptom precaution data loaded successfully.")
except Exception as e:
    print(f"❌ Error loading precaution CSV: {e}")
    precaution_df = None

# Skin Disease (DL) Model and Data - CORRECTED SECTION
try:
    # a. Define the model architecture with 7 classes
    skin_model = models.efficientnet_b0(weights=None)
    skin_model.classifier[1] = nn.Linear(skin_model.classifier[1].in_features, 7)
    skin_model.load_state_dict(torch.load(os.path.join(basedir, 'models/skin_model.pth'), map_location=torch.device('cpu')))
    skin_model.eval()
    print("✅ Skin disease model loaded successfully.")

    # b. Define the class names to EXACTLY match your working check code
    skin_class_names = ['AD', 'CD', 'EC', 'OOD', 'SC', 'SD', 'TC']
    
    # c. Map abbreviations to full disease names
    disease_map = {
        "AD": "Atopic Dermatitis", "CD": "Contact Dermatitis", "EC": "Eczema",
        "SC": "Scabies", "SD": "Seborrheic Dermatitis", "TC": "Tinea Corporis",
        "OOD": "No disease predicted! (Check again, photo may be inappropriate)"
    }

    # d. Define image transformations
    skin_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # e. Define precautions using the abbreviations as keys
    skin_precautions_map = {
        "AD": ["Consult a dermatologist and keep the area clean."],
        "CD": ["Avoid irritants and apply soothing creams."],
        "EC": ["Keep skin moisturized and avoid scratching."],
        "SC": ["Use medicated cream and wash clothes properly."],
        "SD": ["Apply antifungal creams and keep skin dry."],
        "TC": ["Consult a doctor and follow treatment instructions."]
    }
    print("✅ Skin disease model configuration is ready.")

except Exception as e:
    print(f"❌ Error loading skin disease model: {e}")
    skin_model = None


# --- 3. HELPER FUNCTIONS ---

def predict_disease_from_symptoms(symptoms_list):
    # This function is unchanged
    if not symptom_model or precaution_df is None: return {"error": "Server-side models or data are not loaded."}
    processed_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms_list]
    valid_symptoms = [s for s in processed_symptoms if s in symptom_encoder.classes_]
    if not valid_symptoms: return {"error": "No valid symptoms provided or symptoms are not recognized by the model."}
    input_vector = pd.DataFrame(0, index=[0], columns=symptom_encoder.classes_)
    for symptom in valid_symptoms: input_vector[symptom] = 1
    predicted_disease_index = symptom_model.predict(input_vector)[0]
    confidence = max(symptom_model.predict_proba(input_vector)[0]) * 100
    predicted_disease_name = label_encoder.inverse_transform([predicted_disease_index])[0]
    precautions_row = precaution_df[precaution_df['Disease'] == predicted_disease_name]
    precautions = precautions_row.iloc[0, 1:].dropna().tolist() if not precautions_row.empty else ["No specific precautions found."]
    return {"disease": predicted_disease_name, "accuracy": f"{confidence:.2f}%", "precautions": precautions}

def get_remedies_from_bot(disease_name):
    # This function is unchanged
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"You are a friendly health assistant. Suggest 3 to 5 simple home remedies for '{disease_name}'. Keep instructions clear, simple, and easy to follow. Format as a simple text list without intro/conclusion. Do not use markdown."
        response = model.generate_content(prompt)
        remedies_text = response.text.replace('*', '').replace('•', '').strip()
        follow_up_question = "\n\nIs there any other disease I can help you with?"
        return remedies_text + follow_up_question
    except Exception as e:
        print(f"ERROR during Gemini API call for remedies: {e}")
        return "Sorry, I couldn't fetch remedies at the moment. Please try again later."

def get_chatbot_response(user_message):
    # This function is unchanged
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f'You are MedTed, a professional and empathetic AI health assistant. Respond clearly, politely, and concisely. User says: "{user_message}". Give helpful general health information, but strictly avoid giving medical advice or diagnoses.'
        response = model.generate_content(prompt)
        return response.text.replace('*', '').strip()
    except Exception as e:
        print(f"ERROR during Gemini API call for chatbot: {e}")
        return "Sorry, I'm having trouble connecting right now. Please try again later."

def predict_skin_disease(image_bytes):
    """
    Predicts a skin disease using the corrected class names and mappings.
    """
    if not skin_model:
        return {"error": "Skin model is not loaded on the server."}
        
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = skin_transforms(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = skin_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
        
        predicted_class_abbr = skin_class_names[pred_idx.item()]
        confidence_score = f"{confidence.item() * 100:.2f}%"

        # Convert abbreviation to full disease name
        disease_name = disease_map.get(predicted_class_abbr, "Unknown Prediction")
        
        # Get precautions based on the predicted abbreviation
        if predicted_class_abbr == "OOD":
            precautions = "Nothing to be display"
        else:
            precautions = skin_precautions_map.get(predicted_class_abbr, ["No specific precautions found."])

        return {"disease": disease_name, "confidence": confidence_score, "precautions": precautions}

    except Exception as e:
        print(f"ERROR during image prediction: {e}")
        return {"error": "An error occurred while processing the image."}


# --- 4. FLASK ROUTES ---
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
@app.route('/<string:page_name>')
def render_page(page_name='index.html'):
    return render_template(page_name)

@app.route('/api/predict_symptoms', methods=['POST'])
def api_predict_symptoms():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        result = predict_disease_from_symptoms(symptoms)
        return jsonify(result)
    except Exception as e:
        print(f"SERVER ERROR during symptom prediction: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/get_remedies', methods=['POST'])
def api_get_remedies():
    try:
        data = request.get_json()
        disease_name = data.get('disease', '')
        if disease_name.lower() in ['no', 'nO', 'No', 'NO']:
            return jsonify({"remedies": "Wish you for a better health! Take care"})
        remedies_text = get_remedies_from_bot(disease_name)
        return jsonify({"remedies": remedies_text})
    except Exception as e:
        print(f"SERVER ERROR during remedies fetching: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/predict_skin', methods=['POST'])
def api_predict_skin():
    try:
        if 'image' not in request.files: return jsonify({"error": "No image file provided."}), 400
        file = request.files['image']
        if file.filename == '': return jsonify({"error": "No image selected for upload."}), 400
        img_bytes = file.read()
        result = predict_skin_disease(img_bytes)
        return jsonify(result)
    except Exception as e:
        print(f"SERVER ERROR during skin prediction: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        if not message: return jsonify({"response": "Please type a message."})
        bot_response = get_chatbot_response(message)
        return jsonify({"response": bot_response})
    except Exception as e:
        print(f"SERVER ERROR during chat: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- 5. RUN THE APPLICATION ---
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render sets this automatically
    app.run(debug=True, host='0.0.0.0', port=port)


