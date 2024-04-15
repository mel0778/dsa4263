# Insider Threat Detection 

### Description
This project aims to detect potential insider threats within an organization using machine learning techniques. Insider threats can come from current or former employees, contractors, or partners who have access to sensitive information and may misuse it. Our approach uses a synthetic dataset containing labeled insider threat activities to train models that can identify malicious behavior based on anomalies in user data such as login times, file accesses, and other system activities.

### Key Objectives
1. Minimizing False Positives (FP): Reduce unnecessary alerts and disruptions for normal staff.
Our aim is to optimize the trade-off between True Positives (TP), False Negatives (FN) and
FP, thus improving the overall efficiency of the detection system.

2. Identifying Malicious Users in the following scenario: User begins surfing job websites and
soliciting employment from a competitor. Before leaving the company, they use a thumb
drive (at markedly higher rates than their previous activity) to steal data. 

3. Enhancing Explainability: We elucidate our detection system via post model evaluation and
incorporating metrics like confidence scores. This will enable stakeholders to understand
the model’s decision-making process, particularly crucial for sensitive tasks like insider
threat detection.

### Results and Findings
We constructed a novel iterative ensemble method that leverages the use of confidence scoring metrics to generate final predictions unlike most industry solutions, which only flag and provide a binary outcome regarding malicious behaviour. This yields a new paradigm regarding such problems as we can dynamically readjust FP, FN and TP via tuning a ensemble and threshold parameter.

### Git Repository Structure
```
.
├── data
│   ├── processed
│   ├── raw_external
│   └── materials
│       └── reports
│           └── figures
│           └── Report.pdf
├── models
├── notebooks
│   ├── 1_EDA_RawData.ipynb
│   ├── 2A_Feature_Extract.ipynb
│   ├── 2B_EDA_Post_FE.ipynb
│   ├── 3A_model_DT.ipynb
│   ├── 3B_model_DT_smote.ipynb
│   ├── 3C_model_SVM.ipynb
│   ├── 3D_model_SVM_smote.ipynb
│   ├── 3E_model_NN.ipynb
│   ├── 3F_model_NN_smote.ipynb
│   ├── 3G_Ensemble_Model.ipynb
│   ├── 4_Model_Comparisons.ipynb
│   └── 5_EDA_Post_Models.ipynb
└── python_scripts
    └── data
        └── make_dataset.py

```
### Installing Dependencies

#### For Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### For macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
#### Installing packages
```bash
pip install -r requirements.txt
```

#### Docker Setup

For instructions on using Docker to run our models and export model weights efficiently, please see the README in the `docker` branch. This branch includes specialized Docker configurations for our project tasks.

### Usage
1. Automates raw external dataset 
- Automates dataset generation by downloading and extracting files from https://kilthub.cmu.edu/ndownloader/files, organizing them into /data/processed/raw_external folder.
```
cd /dsa4263/python_scripts/data 
python make_dataset.py
```
2. Run notebooks in /dsa4263/notebooks
1_EDA_RawData.ipynb - Initial exploratory data analysis on the raw data
2A_Feature_Extract.ipynb - Notebook for feature extraction from the data
2B_EDA_Post_FE.ipynb - Exploratory data analysis on post feature extraction data
3A_model_DT.ipynb - Decision Tree model
3B_model_DT_smote.ipynb - Decision Tree model with SMOTE oversampling
3C_model_SVM.ipynb - Support Vector Machine model
3D_model_SVM_smote.ipynb - SVM model with SMOTE oversampling
3E_model_NN.ipynb - Neural Network model
3F_model_NN_smote.ipynb - Neural Network with SMOTE oversampling
3G_Ensemble_Model.ipynb - Ensemble model combining various models
4_Model_Comparisons.ipynb - Comparisons between different models 
5_EDA_Post_Models.ipynb - EDA after model building

