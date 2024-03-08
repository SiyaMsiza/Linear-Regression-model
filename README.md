# Linear Regression Project

Linear Regression is a fundamental algorithm in machine learning for predicting a quantitative response. It models the relationship between a dependent variable and one (simple linear regression) or more (multiple linear regression) independent variables by fitting a linear equation to observed data. The core idea is represented by the equation **Y = wX + b**.

## Key Components

- **Y**: Dependent Variable (Target)
- **X**: Independent Variable (Feature)
- **w**: Weight (Coefficient)
- **b**: Bias (Intercept)

## Gradient Descent

An optimization algorithm used to minimize the loss function by iteratively moving toward the minimum loss by updating the model's parameters (weights and biases).

## Learning Rate

A hyperparameter that controls how much we are adjusting the weights of our network with respect to the loss gradient. It determines the size of the steps we take during optimization.

## Implementation

This project implements a simple linear regression model using Python and NumPy. Below is a guide on setting up and running the project.

### Prerequisites

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn (Optional for advanced visualizations)
- sklearn (For train_test_split utility)

### Import the Required Library

```python
import numpy as np

# Linear Regression Class

class Linear_Regression:
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape  # m = Number of examples, n = Number of features
        self.w = np.zeros(self.n)  # Weight initialization
        self.b = 0  # Bias initialization
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_prediction = self.predict(self.X)
        dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = -(2 * np.sum(self.Y - Y_prediction)) / self.m
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b
```
## Project Workflow

### Data Preparation:
Load and preprocess the data.
Split it into training and testing sets.

### Model Training:
Initialize the Linear Regression model.
Train the model on the training data.

### Evaluation:
Predict on the testing set.
Visualize the model performance.

## Implementation Steps

Load and Visualize the Data

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset
salary_data = pd.read_csv("path/to/salary_data.csv")

# Visualize the first few rows
salary_data.head()

# Check for missing values
salary_data.isnull().sum()

# Preprocessing and Splitting
X = salary_data[['YearsExperience']]
Y = salary_data['Salary'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

# Model Training

model = Linear_Regression(learning_rate=0.02, no_of_iterations=1000)
model.fit(X_train, Y_train)
```

# Evaluation and Visualization

```
# Predict on testing data
predictions = model.predict(X_test)

# Visualize predictions

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, predictions, color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Years of Experience')
plt.show()
```
# Conclusion
This simple Linear Regression project demonstrates how to predict a dependent variable using one independent variable. The model's performance can be evaluated visually by comparing the actual and predicted values, providing insights into the effectiveness of the model.

# Enhancements
Experiment with different learning rates and iterations.
Implement additional features such as regularization to prevent overfitting.
Explore other optimization algorithms besides gradient descent.
