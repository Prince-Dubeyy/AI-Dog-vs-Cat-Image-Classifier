import React, { useState, useRef, useCallback } from 'react';
import { Upload, Image as ImageIcon, Loader2, CheckCircle2, AlertCircle, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import './index.css';

interface PredictionResult {
  prediction: string;
  confidence: number;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file (JPEG, PNG, etc.)');
      return;
    }
    setError(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null);
  };

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  const clearImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || '';
      const response = await axios.post(`${apiUrl}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || 'Failed to analyze the image. Is the backend running?');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getConfidenceClass = (confidence: number) => {
    if (confidence >= 90) return 'high-conf';
    if (confidence >= 70) return 'med-conf';
    return 'low-conf';
  };

  return (
    <div className="app-container">
      {/* Navbar */}
      <nav className="navbar">
        <div className="logo">
          <div className="logo-icon">
            <Sparkles size={20} />
          </div>
          <span>VisionPro</span>
        </div>
      </nav>

      <main className="main-content">
        {/* Hero Section */}
        <div className="hero">
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="hero-title"
          >
            AI <span className="gradient-text">Dog vs Cat</span> Classifier
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="hero-subtitle"
          >
            Experience state-of-the-art computer vision. Upload an image and let our deep learning model accurately identify whether it's a dog or a cat.
          </motion.p>
        </div>

        {/* Grid Layout */}
        <div className="grid-container">
          
          {/* Left Column: Upload Section */}
          <div className="glass-panel">
            <div 
              className={`upload-area ${isDragging ? 'dragging' : ''} ${previewUrl ? 'has-preview' : ''}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => !previewUrl && fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
                style={{ display: 'none' }}
                accept="image/*"
              />

              {!previewUrl ? (
                <>
                  <div className="upload-icon">
                    <Upload size={32} />
                  </div>
                  <h3 className="upload-title">Drag & Drop Image</h3>
                  <p className="upload-subtitle">or click to browse your files</p>
                  <button className="btn btn-primary">
                    Select Image
                  </button>
                </>
              ) : (
                <div className="preview-container">
                  <img src={previewUrl} alt="Preview" className="preview-image" />
                  <div className="preview-overlay">
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        fileInputRef.current?.click();
                      }}
                      className="btn btn-outline"
                    >
                      Change Image
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Error Message */}
            <AnimatePresence>
              {error && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="error-message"
                >
                  <AlertCircle size={20} />
                  <span>{error}</span>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Action Buttons */}
            {previewUrl && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="actions"
              >
                <button
                  onClick={analyzeImage}
                  disabled={isAnalyzing}
                  className="btn btn-primary"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 size={20} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <CheckCircle2 size={20} />
                      Analyze Image
                    </>
                  )}
                </button>
                <button
                  onClick={clearImage}
                  disabled={isAnalyzing}
                  className="btn btn-outline"
                >
                  Clear
                </button>
              </motion.div>
            )}
          </div>

          {/* Right Column: Results Section */}
          <div className="glass-panel">
            <AnimatePresence mode="wait">
              {!result && !isAnalyzing ? (
                <motion.div 
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="result-placeholder"
                >
                  <ImageIcon size={64} />
                  <p>Upload an image and hit analyze to see<br/>the AI's prediction here.</p>
                </motion.div>
              ) : isAnalyzing ? (
                <motion.div
                  key="analyzing"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="analyzing-state"
                >
                  <div className="spinner-container">
                    <div className="spinner-ring"></div>
                    <div className="spinner-spin"></div>
                    <div className="spinner-icon">
                      <Sparkles size={24} />
                    </div>
                  </div>
                  <h3>AI is thinking...</h3>
                  <p style={{ color: 'var(--text-muted)' }}>Extracting features & classifying image</p>
                </motion.div>
              ) : result ? (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, y: 20, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  className="result-content"
                >
                  <div className="prediction-header">
                    <div className="prediction-label">Prediction Result</div>
                    <div className="prediction-value">
                      <span className="prediction-emoji" role="img" aria-label={result.prediction.toLowerCase()}>
                        {result.prediction === 'Dog' ? '🐶' : '🐱'}
                      </span>
                      <span className="gradient-text">{result.prediction}</span>
                    </div>
                  </div>
                  
                  <div className="confidence-section">
                    <div className="confidence-header">
                      <span className="confidence-title">Confidence Score</span>
                      <span className={`confidence-value ${getConfidenceClass(result.confidence)}`}>
                        {result.confidence.toFixed(2)}%
                      </span>
                    </div>
                    
                    <div className="progress-track">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence}%` }}
                        transition={{ duration: 1, ease: "easeOut" }}
                        className={`progress-fill ${getConfidenceClass(result.confidence)}-bg`}
                      />
                    </div>

                    <div className="insight-box">
                      <div className={`insight-icon ${getConfidenceClass(result.confidence)}`}>
                        {result.confidence >= 90 ? <CheckCircle2 size={24} /> : <AlertCircle size={24} />}
                      </div>
                      <div>
                        <div className="insight-title">
                          {result.confidence >= 90 ? 'High Confidence' : 'Moderate Confidence'}
                        </div>
                        <div className="insight-desc">
                          {result.confidence >= 90 
                            ? `The neural network is highly certain that this image contains a ${result.prediction.toLowerCase()}.`
                            : `The network leans towards a ${result.prediction.toLowerCase()}, but some features might be ambiguous or unclear.`}
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
