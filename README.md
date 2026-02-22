# signalSafe- AI Powered Real Time Campus Distress Detection System

##signalSafe is a real-time AI-powered security monitoring system that detects potential distress gestures using computer vision and machine learning. The system uses a camera feed, analyzes hand landmarks using MediaPipe, classifies gestures using a trained ML model, and displays alerts in a web-based Security Operations Center (SOC) dashboard. It supports: Real-time distress detection, Live alert, Alert logging & false positive tracking, Escalation to campus safety, ML confidence scoring.

##In large campuses and public spaces, silent distress signals can go unnoticed. signalSafe provides: Automated gesture-based distress detection, Real-time monitoring dashboard, Structured incident logging, Reduced response time.

##Tech Stack 
###Backend: 
Python 
Flask 
OpenCV 
MediaPipe 
Scikit-learn 
Joblib 
###Frontend: 
HTML 
CSS 
JavaScript 
###Machine Learning: 
Custom trained classifier (distress_classifier.joblib) 
Training data stored in X_hand.npy and y_hand.npy
