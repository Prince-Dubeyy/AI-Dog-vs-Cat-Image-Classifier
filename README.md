# 🐶🐱 AI Dog vs Cat Image Classifier

An AI-powered web application that classifies uploaded images as either **Dog** or **Cat** using a Deep Learning model built with TensorFlow and served through a FastAPI backend.

Users can upload an image through a modern React interface and receive an instant AI prediction with confidence scores.

---

# 🚀 Live Demo

### Frontend (Vercel)

https://ai-dog-vs-cat-image-classifier.vercel.app/

### Backend API (Render)

https://ai-dog-vs-cat-image-classifier-1.onrender.com

---

# ✨ Features

- AI-powered Dog vs Cat image classification
- Deep Learning model using TensorFlow/Keras
- FastAPI REST API
- Modern React + Vite frontend
- Drag & Drop image upload
- Responsive UI
- Real-time prediction with confidence score
- Production deployment using Vercel & Render

---

# 🛠️ Tech Stack

### Frontend

- React
- Vite
- TypeScript
- Tailwind CSS
- Axios

### Backend

- Python
- FastAPI
- TensorFlow
- Keras
- Uvicorn
- Pillow
- NumPy

### Deployment

- Vercel
- Render

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

# 🧠 Model Information

- Framework: TensorFlow / Keras
- Task: Binary Image Classification
- Classes:
  - 🐶 Dog
  - 🐱 Cat

---

# 🌐 API Endpoint

## POST `/predict`

Upload an image and receive:

```json
{
  "prediction": "Dog",
  "confidence": 99.42
}
```

---

# 📸 Screenshots

## Home Page

(Add your screenshot here)

## Prediction Result

(Add your screenshot here)

---

# 🚀 Deployment

| Service | Platform |
|---------|----------|
| Frontend | Vercel |
| Backend | Render |

---

# 🔮 Future Improvements

- Multi-class animal classification
- Mobile camera support
- Batch image prediction
- Prediction history
- User authentication
- Model retraining pipeline

---

# 👨‍💻 Author

**Prince Dubey**

B.Sc. Data Science & Cyber Security

📧 princeekjmar@gmail.com

GitHub:
https://github.com/Prince-Dubeyy

---

# ⭐ If you found this project interesting, consider giving it a Star!
