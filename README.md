# Distance-Based Classification: 1-NN on the Sonar Dataset (Rock vs. Metal Data)

This project implements a **1-Nearest Neighbor (1-NN)** classifier from scratch and applies it to the classic **Sonar Rock vs Metal Cylinder dataset**. Each instance contains 60 continuous features representing sonar signal energy patterns, and the goal is to classify whether the detected object is:

- **R** — Rock  
- **M** — Metal Cylinder

The classifier uses the **Minkowski distance metric**, evaluated under two settings:

- **q = 1** → Manhattan distance  
- **q = 2** → Euclidean distance  

For each test instance, the algorithm finds the closest training example and assigns its label.  
The model is evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

---

## Dataset

The dataset consists of:

- **sonar_train.csv** — 139 samples × 60 features  
- **sonar_test.csv** — 69 samples × 60 features  
- **Class labels:**  
  - **R** (Rock)  
  - **M** (Metal Cylinder)

Source: UCI Machine Learning Repository.

---

## Methodology

### **1. Custom Minkowski Distance Function**
Distance is computed manually using:

\[
d(x, y) = \left(\sum |x_i - y_i|^{q}\right)^{1/q}
\]

### **2. Manual 1-NN Classifier**
For each test sample:
- compute distances to all training samples  
- select the nearest one  
- assign its class label  

### **3. Evaluation Metrics**
Performance is assessed using the metrics for the positive class **“M”**:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix (heatmap)

---

## Results

| Distance Metric | Accuracy | Precision | Recall | F1-Score |
|-----------------|----------|-----------|--------|---------|
| **Manhattan (q = 1)** | 0.884 | 0.854 | 0.946 | 0.897 |
| **Euclidean (q = 2)** | 0.899 | 0.857 | 0.973 | 0.911 |

**Summary:**  
Both distance metrics perform strongly, but **Euclidean distance slightly outperforms Manhattan distance** on this dataset.  
This suggests that the geometry of the sonar feature space aligns well with L2-norm similarity.

---

## Skills Demonstrated

- Manual implementation of machine-learning algorithms  
- Minkowski, Manhattan, and Euclidean distance metrics  
- Supervised classification  
- Data preprocessing and numeric feature handling  
- Evaluation using accuracy, precision, recall, F1  
- Confusion matrix visualisation with seaborn  
- Working with UCI datasets in Python  

---

## Files in This Repository

- `1-NN_Classifier_on_Sonar_Dataset.ipynb` — main notebook  
- `sonar_train.csv` — training data  
- `sonar_test.csv` — test data  

---

## Summary

This project highlights the fundamentals of distance-based classification and shows how even a simple 1-NN algorithm can effectively separate classes in a high-dimensional numerical dataset. It serves as a clear demonstration of essential machine-learning concepts implemented from first principles.

