# AI Dog vs Cat Image Classifier 🐶🐱

An enterprise-grade, high-performance deep learning web application that accurately classifies images as either a Dog or a Cat. 

This project features a powerful **MobileNetV2 Transfer Learning** backend model and a stunning, fully responsive **Vanilla CSS** frontend with modern glassmorphism UI/UX.

## ✨ Features

- **High Accuracy AI Model**: Powered by a MobileNetV2 architecture trained on the `oxford_iiit_pet` dataset, achieving **98.09% Validation Accuracy**.
- **Real-time Inference**: Fast and accurate predictions using TensorFlow and a FastAPI backend.
- **Premium Frontend UI**: Zero CSS-framework dependencies! Built entirely with Vanilla CSS featuring:
  - Frosted Glassmorphism aesthetics
  - Dynamic gradient backgrounds
  - Smooth micro-animations powered by Framer Motion
  - Fully responsive grid layout for Mobile, Tablet, Laptop, and Desktop.
- **Intelligent Confidence Scoring**: Proportional and mathematically sound confidence mapping for true prediction transparency.

## 🛠️ Tech Stack

### Backend
- **Python 3**
- **TensorFlow / Keras** (MobileNetV2 Transfer Learning)
- **FastAPI & Uvicorn** (RESTful API)
- **Pillow & Numpy** (Image preprocessing)

### Frontend
- **React.js & TypeScript**
- **Vite** (Build tool)
- **Vanilla CSS** (Styling)
- **Framer Motion & Lucide React** (Animations and Icons)
- **Axios** (API Client)

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Prince-Dubeyy/AI-Dog-vs-Cat-Image-Classifier.git
cd AI-Dog-vs-Cat-Image-Classifier
```

### 2. Set up the Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
The backend API will run on `http://127.0.0.1:8000`.

### 3. Set up the Frontend
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
The application will be accessible at `http://localhost:5173`.

*Note: The frontend development script `npm run dev` is configured to run both the Vite server and the FastAPI backend concurrently for convenience.*

## 🧠 Model Training

To retrain the model or fine-tune it yourself:
```bash
cd backend
python train.py
```
This script will automatically download the `tensorflow_datasets` and run a 2-phase training cycle (Initial Training + Fine-Tuning) with EarlyStopping and learning rate adjustments.

## 🧪 Validation

To test the model against live unseen data from the internet:
```bash
cd backend
python validate.py
```
This fetches 10 random cat and 10 random dog images from live APIs, processes them through the CNN, and prints a comprehensive classification report and confusion matrix.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License
[MIT](https://choosealicense.com/licenses/mit/)
