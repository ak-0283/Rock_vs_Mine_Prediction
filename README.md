# Rock vs Mine Prediction using Machine Learning

This is my **first Machine Learning project** after learning from [Siddhardhan's YouTube channel](https://www.youtube.com/@Siddhardhan) 🚀.
The project predicts whether the object is a **Rock (R)** or a **Mine (M)** using Logistic Regression.

---

## 📚 Learning Journey

* I followed Siddhardhan's tutorials on YouTube to understand **Machine Learning fundamentals**.
* Implemented my learnings using **Google Colab**.
* Explored data preprocessing, model training, and evaluation.

---

## 🔁 Project Workflow

1. **Data Loading** using pandas  
2. **Exploratory Data Analysis**  
3. **Label Encoding** of categorical target labels  
4. **Train-Test Split** using `train_test_split`  
5. **Model Building** using Logistic Regression  
6. **Model Evaluation** with accuracy on training and testing sets

---

## 📂 Dataset Information

* Dataset contains **208 rows** and **61 columns**.
* Features: 60 independent variables representing sonar signals.
* Target:

  * `M` → Mine (111 samples)
  * `R` → Rock (97 samples)

> **Note**: The dataset is  included in this repository in the dataset.txt file. Please download it manually or else you download  dataset from any website like kaggle or uci and then can you check it 👍.
---

## 🛠️ Libraries Used

- Python
- Google Colab
- NumPy
- Pandas
- scikit-learn (Logistic Regression, LabelEncoder, train_test_split, accuracy_score)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## 📊 Data Splits

* Total samples: **208**
* Training samples: **187**
* Testing samples: **21**

```python
X.shape       # (208, 60)
X_train.shape # (187, 60)
X_test.shape  # (21, 60)
```

---

## 🤖 Model Training & Accuracy

* **Algorithm:** Logistic Regression
* **Training Accuracy:** 83.42%
* **Testing Accuracy:** 76.19%

```python
print('Accuracy on training data : ', training_data_accuracy)
# Accuracy on training data :  0.8342245989304813

print('Accuracy on test data : ', test_data_accuracy)
# Accuracy on test data :  0.7619047619047619
```

---

## 💻 How to Run

1. Clone this repository

```bash
git clone https://github.com/your-username/rock-vs-mine-ml.git
```

2. Install dependencies

```bash
pip install numpy pandas scikit-learn
```

3.Run the notebook in **Google Colab** using the badge above or open locally:

```bash
jupyter notebook notebooks/ML Use Case 1. Rock_vs_Mine_Prediction.ipynb
```

---

## 🎯 Conclusion

This project helped me:

* Understand **data preprocessing**
* Learn **Logistic Regression** basics
* Evaluate **model performance**

It was a great learning experience and my first step into the world of **Machine Learning**! 🚀. ⭐️ If you found this helpful, consider giving this repo a star!

---

## 🙌 Acknowledgments

Special thanks to **Siddhardhan** for his clear and beginner-friendly tutorials on YouTube.

---
