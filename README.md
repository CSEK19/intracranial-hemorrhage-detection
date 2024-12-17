# Lightweight Classifier for Detecting Intracranial Hemorrhage in Ultrasound Data

This project focuses on developing a lightweight machine learning classifier for detecting intracranial hemorrhage using ultrasound Tissue Pulsatility Imaging (TPI) data. The goal is to create a reliable, efficient, and portable solution for real-time diagnosis in diverse environments, such as rural areas or emergency settings.

## **Folder Structure**

### **Dataset**
Contains preprocessed datasets in MATLAB format:
- `original.mat`: Full dataset with all features (raw data with 30 time frames and recording angle as features).
- `reduced.mat`: Dataset after feature selection, retaining only the most significant features.
- `transformed.mat`: Dataset after applying Principal Component Analysis (PCA), transformed into a new space defined by principal components.

### **Outputs**
Stores results and performance metrics:
- **Text Files**:
  - `Output_original.txt`: Results for models trained on the original dataset.
  - `Output_reduced.txt`: Results for models trained on the reduced dataset.
  - `Output_transformed.txt`: Results for models trained on the transformed dataset.
- **Excel Files**:
  - `ResultsTable_original.xlsx`: Detailed metrics for the original dataset.
  - `ResultsTable_reduced.xlsx`: Detailed metrics for the reduced dataset.
  - `ResultsTable_transformed.xlsx`: Detailed metrics for the transformed dataset.

### **Scripts**
MATLAB scripts for training and evaluating models:
- `AdaBoostModel.m`: AdaBoost classifier.
- `GaussianModel.m`: Gaussian Naive Bayes classifier.
- `LogitBoostModel.m`: LogitBoost ensemble classifier.
- `NNModel.m`: Neural Network model.
- `RUSBoostModel.m`: RUSBoost ensemble classifier.
- `SVMModel.m`: Support Vector Machine classifier.

### **Documentation**
- `Paper.pdf`: Research paper detailing methodology, results, and future work.

## **How to Use**

### **Step 1: Data Preparation**
1. Load a `.mat` file from the `Dataset` folder into MATLAB: `load('Dataset/original.mat');`
2. Perform additional preprocessing if needed (e.g., normalization).

### **Step 2: Model Training**
Run any script to train a specific model. For example: `run('AdaBoostModel.m');`

### **Step 3: Results Analysis**
Review the Excel files in the `Outputs` folder to compare performance metrics such as accuracy, precision, recall, and F1-score across datasets and models.

## **Key Features of Each Script**

| Script Name         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `AdaBoostModel.m`   | Trains an AdaBoost ensemble classifier optimized for imbalanced datasets.   |
| `GaussianModel.m`   | Implements Gaussian Naive Bayes as a baseline model.                        |
| `LogitBoostModel.m` | Trains a LogitBoost ensemble model for logistic regression classification.  |
| `NNModel.m`         | Builds a Neural Network for non-linear relationships in high-dimensional data. |
| `RUSBoostModel.m`   | Uses random undersampling to address class imbalance in ensemble boosting.  |
| `SVMModel.m`        | Trains an SVM classifier with kernel methods for linear/non-linear tasks.  |

---

## **Performance Summary**

Results are summarized across three datasets:
1. **Original Dataset**: Full feature set (30 time frames + recording angle).
2. **Reduced Dataset**: Selected features after loading vector analysis (most significant features retained).
3. **Transformed Dataset**: Principal components from PCA, representing transformed data.

| Model            | Best Accuracy (%) | Best F1-score (%) | Dataset Used       |
|------------------|-------------------|-------------------|--------------------|
| AdaBoost         | 98.0             | 88.1             | Transformed        |
| LogitBoost       | 98.0             | 87.7             | Transformed        |
| RUSBoost         | 97.9             | 89.0             | Transformed        |
| Neural Network   | 97.0             | 83.5             | Transformed        |
| SVM              | 82.5             | 62.2             | Transformed        |
| Gaussian NB      | 81.1             | 53.0             | Transformed        |

---

## **Future Work**
1. Incorporate raw ultrasound image data alongside TPI signals for multimodal analysis.
2. Explore advanced dimensionality reduction techniques like ICA.
3. Investigate frequency-domain transformations (e.g., FFT) to enhance feature extraction and model accuracy.

For detailed insights into methodology and results, refer to `Paper.pdf`.

---

## **Contact Information**

For questions or contributions, please contact:
- Fred Xu: zxu725@uw.edu
- Enbai Kuang: enbaik@uw.edu
- Phat Tran: phattt@uw.edu
