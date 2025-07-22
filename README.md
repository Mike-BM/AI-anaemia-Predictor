# 🩸 AI Anemia Predictor


This is an AI-powered web app that predicts whether a person is anemic based on simple health metrics. Built using `Streamlit`, `Scikit-learn`, and `Python`, the tool provides fast, accessible screening support — especially for low-resource healthcare settings.

---

## 📌 Project Summary

- **Goal:** Early detection of anemia using ML classification
- **ML Model:** Trained using patient features like hemoglobin, RBC count, age, gender, and fatigue
- **Deployment:** Live on Streamlit Cloud
- **SDG Alignment:** [SDG 3 – Good Health and Well-being](https://sdgs.un.org/goals/goal3)

---

## 🚀 Try It Live

👉 [Click here to open the app](https://ai-anaemia-predictor-7jgzkpcq3zztfqdg7b2cgt.streamlit.app/)

---

## 🛠 Features

- 🔍 Interactive input form for individual prediction
- 📥 CSV upload for batch anemia detection *(optional enhancement)*
- 📊 Real-time prediction results with health tips
- ✅ Lightweight & fast: suitable for low-resource environments

---

## 📂 Project Structure

```bash
anemia-predictor/
│
├── app.py                # Streamlit frontend
├── anemia_model.pkl      # Trained ML model
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
| file_.csv               # dataset



# Clone the repo
git clone https://github.com/yourusername/anemia-predictor.git
cd anemia-predictor

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
