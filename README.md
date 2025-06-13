
# 😷 Face Mask Detection – AI Project

This project is an AI-based real-time face mask detection system built using **Python**, **TensorFlow/Keras**, and **OpenCV**. It was developed during the **"EDUNET Foundation - Microsoft - AI Azure 4-Week Internship Program"**.

---

## 📌 Features

- Real-time face detection using webcam
- Classification: Mask / No Mask using a trained model
- Model training and testing included
- Uses Haar Cascade for face detection

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Jupyter Notebook
- NumPy

---

## 📂 Folder Structure

```
Face-Mask-Detection/
├── MyTrainingModel.h5                # Trained Keras model for mask detection
├── PreprocessingAndTraining.ipynb   # Jupyter Notebook for training the model
├── README.md                         # Project documentation (you are here)
├── dataCollector.py                 # Script to collect face data using webcam
├── haarcascade_frontalface_default.xml  # Haar cascade for face detection
├── images.zip                        # Collected face images (With Mask/Without Mask)
├── test.py                           # Real-time detection using webcam
```

---

## 🚀 How to Run

### Step 1: Install required libraries

```bash
pip install tensorflow keras opencv-python numpy
```

### Step 2: Collect face data (if not already done)

```bash
python dataCollector.py
```

### Step 3: Train the model

Use the Jupyter Notebook: `PreprocessingAndTraining.ipynb`

### Step 4: Run real-time detection

```bash
python test.py
```

---

## 🎓 Developed Under

**EDUNET Foundation – Microsoft AI-Azure Virtual Internship (4 Weeks)**

---

## 👤 Author

- **Prateek Yadav**
- GitHub: https://github.com/pra9536/MicroSoft-AI-Internship
- Email:  prateek9530@gmail.com

---

## 📘 License

This project is open-source and available for educational use.
