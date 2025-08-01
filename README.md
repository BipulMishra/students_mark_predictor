Student Marks Predictor
📖 Project Overview
This project is a classic machine learning implementation that predicts a student's final mark based on two key features: the number of hours they studied and their score on a previous test. It serves as an excellent introduction to the complete machine learning workflow, from data preprocessing to model evaluation and prediction.

The core of the project is a Linear Regression model built using Python and the powerful Scikit-learn library.

✨ Features
Data-driven Prediction: Utilizes historical data to forecast future outcomes.

Feature Scaling: Implements StandardScaler to normalize features, ensuring fair model training.

Train-Test Split: Properly partitions data to prevent information leakage and allow for unbiased evaluation.

Comprehensive Evaluation: Measures model performance using three standard regression metrics:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R-squared (R²) Score

Real-world Application: Includes a function to predict the mark for a new, unseen student.

🛠️ Technologies Used
Python 3.x

Pandas: For data manipulation and creating the initial dataset.

NumPy: For numerical operations, especially for handling the new student data.

Scikit-learn: The core machine learning library used for:

train_test_split

StandardScaler

LinearRegression

mean_absolute_error, mean_squared_error, r2_score

⚙️ How It Works
The project follows a standard and structured machine learning pipeline:

Data Creation: A sample dataset is created using a Pandas DataFrame to simulate real-world student data.

Feature & Target Separation: The data is divided into input features (X: study_hours, previous_score) and the target variable (y: final_mark).

Data Splitting: The dataset is split into an 80% training set (for the model to learn from) and a 20% testing set (for evaluation).

Preprocessing (Scaling): The numerical features (X_train, X_test) are scaled using StandardScaler. This is crucial for ensuring that features with larger ranges don't disproportionately influence the model.

Model Training: A LinearRegression model is instantiated and trained on the scaled training data (X_train_scaled, y_train).

Prediction: The trained model makes predictions on the scaled test data.

Evaluation: The model's predictions are compared against the actual marks from the test set (y_test) to calculate its performance.

🚀 How to Run the Code
Ensure you have Python and the required libraries installed:

pip install pandas numpy scikit-learn

Save the code from the project into a Python file (e.g., marks_predictor.py).

Run the script from your terminal:

python marks_predictor.py

The script will print the original data, a confirmation of model training, the evaluation results, and the final prediction for a new student.

📊 Evaluation Results
The model demonstrated a very high level of accuracy on the test data:

Mean Absolute Error (MAE): 0.84

On average, the model's prediction was off by only 0.84 marks.

Root Mean Squared Error (RMSE): 0.95

Similar to MAE but penalizes larger errors more. Still indicates a very low error margin.

R-squared (R²) Score: 0.99

The model successfully explains 99% of the variance in the students' final marks, indicating an excellent fit.

These results confirm that the model is highly effective and reliable for this dataset.
