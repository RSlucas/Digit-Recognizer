# Digit Recognizer & Image Classification (CNN)

This repository contains two folders of **Convolutional Neural Networks (CNNs)** applied to different classification tasks.

## 📁 Project Structure

```
Digit-Recognizer/
│
├── digit_recogn/
│   └── source.py
│
├── ai_vs_real/
│   └── source.py
```

---

## 🔹 1. Pixel CNN (Digit Recognizer)

* CNN model applied to **pixel-based data (numerical vectors)**
* Does not use `.jpg` images, but preprocessed pixel values
* Similar to classic digit recognition problems (e.g. MNIST)

**Features:**

* Input: pixel vectors
* Model: simple CNN
* Task: digit classification

---

## 🔹 2. Image CNN (Real vs AI Art)

* CNN model applied to **real images (.jpg)**
* Classifies images into:

  * `0` → real art
  * `1` → AI-generated art

**Features:**

* Input: RGB images
* Preprocessing: resizing, augmentations
* Model: multi-layer CNN
* Metric: F1-score

---

## ⚙️ Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* PIL

---

## 🚀 How to Run

1. Install dependencies:

```
pip install torch torchvision pandas pillow scikit-learn
```

2. Run the desired script:

```
python source.py
```

---

## 🧠 Purpose

This repository explores:

* the difference between numerical and image-based inputs
* how CNNs behave in different contexts
* a full pipeline: dataset → model → evaluation

---

## 📌 Note

This code is intended for learning and experimentation purposes and is not fully optimized for production use.

---
