# 🩺 Diabetic Retinopathy Detection System  
**EfficientNetB5 + Transfer Learning + Fine-Tuning + TTA + Grad-CAM**

This project is a deep learning system that detects the severity of **Diabetic Retinopathy (DR)** from retinal fundus images.  
The model is built on **EfficientNetB5** using **transfer learning**, fine-tuned for the task, and enhanced with **Test Time Augmentation (TTA)** to improve prediction stability.  
Additionally, **Grad-CAM** is used to visualize the regions the model focuses on during decision-making, improving interpretability.

---

## 📌 Objective
- Automatically classify diabetic retinopathy into multiple severity stages.
- Provide model interpretability via Grad-CAM visualizations.
- Increase prediction stability with TTA.
- Deliver a user-friendly **Streamlit** web interface.

---


---

## ⚙️ Technologies Used
- **Python 3.10+**
- **TensorFlow / Keras**
- **EfficientNetB5** (`efficientnet.tfkeras`)
- **Albumentations** (data augmentation & TTA)
- **Streamlit** (web UI)
- **OpenCV** (image processing)
- **Grad-CAM** (model interpretability)

---

## 📊 Model Details
- **Base Model:** EfficientNetB5 (pretrained on ImageNet)
- **Output Type:** Regression (mapped to 0–4 classes)
- **Dataset:** APTOS 2019 Blindness Detection
- **Data Augmentation:** Horizontal/vertical flips, rotations, color adjustments
- **Fine-Tuning:** Unfreezing final layers with a low learning rate
- **TTA:** Horizontal flip, vertical flip, 90° rotation, and original image

---

## 🚀 Installation & Running

streamlit run app.py
http://localhost:8501

📈 Performance
Metric	Score
Validation Loss	0.4639
Quadratic Weighted Kappa	0.8372
QWK with TTA	0.8320

📌 Class Definitions
0: No DR
1: Mild DR
2: Moderate DR
3: Severe DR
4: Proliferative DR

🔍 Explainability with Grad-CAM
Grad-CAM highlights the regions of the retina image that most influenced the model’s prediction, aiding medical experts in validating AI-assisted diagnoses.


If you want, I can also add a **"Run on Google Colab with ngrok"** section so users can launch it without local setup.  
That would make your repo beginner-friendly and runnable in the cloud.

