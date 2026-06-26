# 🐶🐱 AI Dog vs Cat Image Classifier

An AI-powered web application that classifies uploaded images as either **Dog** or **Cat** using a Deep Learning model built with TensorFlow and served through a FastAPI backend.

The application provides a modern React frontend where users can upload an image and receive an instant prediction with confidence scores.

---

## 🚀 Live Demo

### Frontend (Vercel)

https://ai-dog-vs-cat-image-classifier.vercel.app/

### Backend API (Render)

https://ai-dog-vs-cat-image-classifier-1.onrender.com

---

# 📌 Features

- Upload Dog or Cat images
- AI-powered image classification
- FastAPI REST API
- TensorFlow/Keras Deep Learning model
- Drag & Drop image upload
- Responsive UI
- CORS-enabled backend
- Production deployment using Render & Vercel

---

# 🛠️ Tech Stack

## Frontend

- React
- Vite
- TypeScript
- Tailwind CSS
- Axios

## Backend

- Python
- FastAPI
- Uvicorn
- TensorFlow
- Keras
- Pillow
- NumPy

## Deployment

- Vercel (Frontend)
- Render (Backend)

---

# 📂 Project Structure

```
AI-Dog-vs-Cat-Image-Classifier
│
├── backend
│   ├── main.py
│   ├── requirements.txt
│   ├── runtime.txt
│   ├── dog_cat_classifier.keras
│   ├── train.py
│   └── validate.py
│
├── frontend
│   ├── src
│   ├── public
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Prince-Dubeyy/AI-Dog-vs-Cat-Image-Classifier.git

cd AI-Dog-vs-Cat-Image-Classifier
```

---

# Backend Setup

```bash
cd backend

python -m venv venv

venv\Scripts\activate      # Windows

pip install -r requirements.txt

uvicorn main:app --reload
```

Backend runs on

```
http://127.0.0.1:8000
```

---

# Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend runs on

```
http://localhost:5173
```

---

# Environment Variables

## Frontend (.env)

```env
VITE_API_URL=http://127.0.0.1:8000
```

Production

```env
VITE_API_URL=https://ai-dog-vs-cat-image-classifier-1.onrender.com
```

---

# API Endpoint

## Predict Image

```
POST /predict
```

Upload an image and receive:

```json
{
  "prediction": "Dog",
  "confidence": 99.42
}
```

---

# Model Information

- Deep Learning CNN Model
- TensorFlow / Keras
- Binary Classification
- Classes:
  - Dog
  - Cat

---

# Deployment

## Frontend

Hosted on Vercel

https://ai-dog-vs-cat-image-classifier.vercel.app/

## Backend

Hosted on Render

https://ai-dog-vs-cat-image-classifier-1.onrender.com

---

# Future Improvements

- Mobile Camera Support
- Multiple Image Prediction
- Prediction History
- Top-3 Confidence Scores
- User Authentication
- Model Retraining Dashboard
- Batch Prediction

---

# Screenshots

Add screenshots here:

```
screenshots/home.png
screenshots/result.png
```

---

# Author

**Prince Dubey**

B.Sc. Data Science & Cyber Security

GitHub

https://github.com/Prince-Dubeyy

Email

princeekjmar@gmail.com

---

# License

This project is licensed under the MIT License.

