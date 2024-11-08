# Personalized Clothing Size Prediction Model

## Description
This readme file provides information about the model.

## Installation
To use the model, follow these steps:
1. Clone the repository.
2. Install the required dependencies.
3. Run the model.

## Usage
To use the model, execute the following command:
```
python model.py
```

## Contributing
Contributions are welcome! Please follow the guidelines outlined in CONTRIBUTING.md.

## License
This project is licensed under the MIT License. See LICENSE.md for more information.

## Misc Details
./Misc folder contains all the models including kmeans and randomforest model on different datasets based on different data 
You can run them to see how they actually work. YOu need to move them into the main directory and then run them in sklearn-env environment to easily run them.

## Detail

## Tech Stack for Clothing Size Prediction Model

### 1. **Programming Language: Python**
   - Python was chosen for its extensive libraries, simplicity, and versatility in handling data science and machine learning tasks.

### 2. **Libraries and Frameworks:**

   - **Pandas**: 
     - Used for data manipulation and analysis. Pandas is essential for handling and processing the dataset, particularly for tasks such as data cleaning, transforming, and structuring.

   - **NumPy**: 
     - Provides support for large multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. NumPy is critical for performing numerical operations and ensuring efficient computations.

   - **Scikit-learn**:
     - The core library used for implementing machine learning algorithms. It provides simple and efficient tools for data mining, data analysis, and machine learning. Key functionalities used include:
       - **K-Means Clustering**: Applied for initial exploratory analysis to group data points into clusters based on similarity.
       - **Random Forest Classifier**: The primary model used for predicting clothing sizes based on the dataset. It operates by building multiple decision trees and merging them to get a more accurate and stable prediction.

   - **Matplotlib** and **Seaborn**:
     - Visualization libraries used for plotting graphs, including the confusion matrix. These libraries help in analyzing and presenting the model's performance and data distribution.

   - **LabelEncoder**:
     - Used to encode categorical features such as Gender. This transformation is necessary for feeding non-numeric data into the machine learning models.

   - **SimpleImputer**:
     - A tool from Scikit-learn used for handling missing values in the dataset. The imputer was configured to replace missing numerical values with the mean of the column, ensuring no data loss during model training.

   - **Joblib**:
     - Utilized for saving the trained Random Forest model to a file. This allows the model to be reloaded and used later without the need to retrain, which is beneficial for deployment on the e-commerce platform.

### 3. **Data Handling:**
   - The dataset consists of features like Gender, Height (in feet and inches), Weight, Bust/Chest, Cup Size, Waist, Hips, and Body Shape Index.
   - Preprocessing steps include converting height from feet-inches to centimeters, encoding categorical variables, and handling missing values. 

### 4. **Model Training:**
   - The model training process uses 80% of the dataset for training and the remaining 20% for testing. This ensures that the model is evaluated on unseen data to measure its real-world performance.
   - **K-Means** is utilized initially for clustering and exploratory data analysis, while **Random Forest** serves as the primary algorithm for the final prediction of clothing sizes.

### 5. **Evaluation:**
   - The model's performance is evaluated using a confusion matrix, which helps in understanding the accuracy and distribution of predictions across different size categories.

### 6. **Deployment:**
   - The trained Random Forest model is saved using Joblib, making it easy to deploy on an e-commerce platform for real-time size recommendations.


This tech stack supports an end-to-end machine learning pipeline, from data preprocessing to model deployment, ensuring that the clothing size prediction model is both accurate and efficient.
