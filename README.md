# MEDiWiSE: AI-Powered Health Assistant

![MEDiWiSE Homepage](static/img/new-logo.png)

**MEDiWiSE** is an intelligent web application designed to provide preliminary health insights using a suite of AI-powered tools. It serves as a user-friendly first step for individuals seeking information about their health, offering features from symptom-based disease prediction to AI-driven chatbots for general health queries.

---

## ✨ Features

This project integrates multiple AI models into a responsive and intuitive web interface. The core features include:

1.  **Symptom-Based Disease Prediction:**
    * Users can input one or more physical symptoms.
    * A pre-trained **Random Forest** machine learning model (`symptom_disease_model_rf.pkl`) analyzes the input.
    * The application predicts a potential disease and provides a list of corresponding precautions sourced from the `Disease precaution.csv` dataset.

2.  **Image-Based Skin Disease Prediction:**
    * Users can upload an image of a skin condition.
    * A pre-trained **EfficientNet-B0** deep learning model (`skin_model.pth`) analyzes the image.
    * The application predicts one of several common skin conditions and provides a list of relevant precautions for the identified condition.
    * Includes a special "Out-of-Distribution" (OOD) class for images that don't match known conditions.

3.  **AI-Powered Home Remedies Bot:**
    * A dedicated page (`remedies.html`) where users can ask for home remedies for a specific ailment (e.g., "Headache").
    * Powered by the **Google Gemini API**, it provides a list of simple, easy-to-follow home remedies.
    * Features a conversational follow-up to enhance user interaction.

4.  **General Health Assistant Chatbot:**
    * An AI chatbot ("MedTed") embedded on every page for general health queries.
    * Also powered by the **Google Gemini API**, it is programmed to be an empathetic assistant that provides helpful information while strictly avoiding medical diagnoses.
    * The UI includes a "MedTed is typing..." indicator and a simulated delay to feel more realistic.

5.  **Responsive User Interface:**
    * The entire website is designed to be fully responsive, adapting to desktops, tablets, and mobile devices.
    * Features a dark-themed, modern design with a slide-out mobile navigation menu.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn, Pandas
* **Deep Learning:** PyTorch, Torchvision
* **Generative AI:** Google Gemini API
* **Frontend:** HTML5, CSS3, JavaScript
* **Environment Management:** Python `venv`, `requirements.txt`
* **Secret Management:** `python-dotenv`

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.8+
* Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
    cd Your-Repo-Name
    ```

2.  **Create and activate a virtual environment:**
    * On Windows:
        ```bash
        python -m venv myenv
        myenv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        python3 -m venv myenv
        source myenv/bin/activate
        ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    * Create a file named `.env` in the root of the project folder.
    * Inside the `.env` file, add your Google Gemini API key in the following format:
        ```
        GEMINI_API_KEY=AIzaSy...your...actual...key...here
        ```

### Running the Application

1.  Make sure your virtual environment is activated.
2.  Run the Flask server from the root directory:
    ```bash
    python app.py
    ```
3.  Open your web browser and navigate to:
    `http://127.0.0.1:5000`

---

## 📁 File Structure

```
.
├── models/
│   ├── symptom_disease_model_rf.pkl
│   ├── skin_model.pth
│   └── ... (encoders)
├── static/
│   ├── css/
│   ├── img/
│   └── js/
├── templates/
│   ├── layout.html
│   ├── index.html
│   ├── ml_predict.html
│   └── ... (other html files)
├── .env
├── .gitignore
├── app.py
└── requirements.txt
```

---

## ⚠️ Disclaimer

This project is for informational and educational purposes only. The predictions and information provided by the AI models and chatbots are **not a substitute for professional medical advice, diagnosis, or treatment**. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.
