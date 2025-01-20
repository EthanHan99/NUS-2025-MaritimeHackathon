# README: Vessel Deficiency Severity Prediction

This project focuses on analyzing, preprocessing, and predicting vessel deficiency severity using natural language processing (NLP) techniques and neural networks. Below is a summary of the workflow and key steps involved:

## Workflow Overview

### 1. Data Exploration
- Load the dataset (`psc_severity_train.csv`) and perform exploratory data analysis (EDA) to understand key features such as:
  - Distribution of `annotation_severity`.
  - Frequency of inspections by `username`.
  - Common deficiency codes (`deficiency_code`).
  - Number of inspections per ship (`PscInspectionId`).
- Visualize the data using bar charts, histograms, and stacked plots for better insights.

### 2. Data Preprocessing
- Extract meaningful text from the `def_text` column using the segment between "Deficiency/Finding:" and "Description Overview:".
- Apply advanced text preprocessing:
  - Lowercasing.
  - Removing special characters and stop words.
  - Applying stemming to reduce words to their root forms.
- Generate text embeddings using the BERT model (`bert-base-uncased`) for semantic representation.
- Normalize the embeddings for better clustering and prediction performance.

### 3. Clustering and Scoring
- Group data by `deficiency_code` and dynamically determine the number of clusters for each group using the silhouette score.
- Apply KMeans clustering to classify deficiency descriptions into subcategories based on embeddings.
- Generate a unique subcategory ID (`deficiency_code-sub_category`).

### 4. Neural Network Prediction
- Use BERT for sequence classification to predict `annotation_severity` based on `def_text_extracted`:
  - Encode labels using `LabelEncoder`.
  - Train-test split the data (80% training, 20% testing).
  - Fine-tune a BERT model (`bert-base-uncased`) using the HuggingFace Trainer API.
  - Evaluate the model's accuracy on the test set.

### 5. Output and Results
- Save the clustering results, including deficiency subcategories, into `psc_comprehensive_analysis.csv`.
- Save EDA findings into `Maritime_Data_exploration.csv`.
- Provide a trained neural network model capable of predicting severity from text descriptions.

## How to Run
1. Install required libraries:
   ```bash
   pip install pandas matplotlib seaborn transformers torch scikit-learn datasets nltk
   ```
2. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```
3. Run the scripts in sequence:
   - `Maritime_annotation_method.ipynb`: Perform EDA and save results.
   - `Maritime_Network_training`:Train and evaluate the BERT-based prediction model.

## Results
- Visualizations of data distributions.
- Subcategorization of deficiency codes for better interpretability.
- A trained BERT model achieving high accuracy in predicting severity levels from textual descriptions.

---
This project combines EDA, NLP, and advanced neural networks to improve vessel deficiency severity prediction effectively.
