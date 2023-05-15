import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, lr=0.01, num_iters=1000):
        self.lr = lr
        self.num_iters = num_iters
        self.w = None
        self.b = None
    
    # D. Implement linear regression from scratch using gradient descent to optimize the parameters of the hypothesis function.
    def fit(self, X, y): # Model
        # Add b column to X
        # X = np.insert(X, 0, 1, axis=1)
        m, features = X.shape
        # Initialize w and b
        self.w = np.zeros(features)
        self.b = 0
        
        # Gradient descent
        costs = []
        for i in range(self.num_iters):
            y_pred = np.dot(X, self.w) + self.b            # Hypothesis function f(x)=wX+b 
            cost = (np.sum((y_pred - y) ** 2))/(2*m)       # J(w,b)=sigma[(f(x)-y)^2]/2m
            dw = (1/m) * np.dot(X.T, (y_pred-y))           # dwJ(w,b)=sigma[f(x)-y]*x/m
            db = (1/m) * np.sum(y_pred-y)                  # dbJ(w,b)=sigma[f(x)-y]/m
            # Simultaneously update w & b
            self.w -= self.lr * dw                         # w=w-alpha*dwJ(w,b)
            self.b -= self.lr * db                         # b=b-alpha*dbJ(w,b)
            
            costs.append(cost)
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")
        
        return self.w, self.b, costs
    
    # H. Use the optimized hypothesis function to make predictions on the testing set 
    def predict(self, X):
        # Add b column to X
        # X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.w) + self.b   # f(x)=wX+b 
    
    # and calculate the accuracy of the final (trained) model on the test set.
    def score(self, X, y):
        y_pred = self.predict(X) # f(x)=wX+b 
        return 1 - (np.sum((y_pred-y) ** 2) / np.sum((y.mean()-y) ** 2))
    

# A. Load the “car_data.csv” dataset.
data = pd.read_csv('car_data.csv')

# Select 7 numerical features and target
num_cols = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'boreratio', 'horsepower', 'price']
data = data[num_cols]

# Drop rows with missing values to solve error of nan
data = data.dropna()

# B. Use scatter plots between different features (7 at least) and the car price
for feature in num_cols[:-1]:
    plt.scatter(data[feature], data['price'])
    plt.xlabel(feature)
    plt.ylabel('Car Price')
    plt.show()

# C. Split the dataset into training and testing sets.
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# select 5 of the numerical features that are positively/negatively correlated to the car price
corr_matrix = train_data.corr()
top_corr_features = corr_matrix['price'].abs().sort_values(ascending=False).head(6).index[1:] # get most correlated features
X_train = train_data[top_corr_features[:-1]].to_numpy()
y_train = train_data[top_corr_features[-1]].to_numpy()
X_test = test_data[top_corr_features[:-1]].to_numpy()
y_test = test_data[top_corr_features[-1]].to_numpy()

# Normalize the data by subtracting the mean and dividing by the standard deviation of the feature matrix.
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Normalize the data by min-max normalization
# X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0)-np.min(X_train, axis=0))
# X_test = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0)-np.min(X_test, axis=0))

# Train the linear regression model
lr = LinearRegression(lr=0.01, num_iters=1000)
w, b, costs = lr.fit(X_train, y_train)

# E. Print the parameters of the hypothesis function.
print("w:", w)
print("b:", b)

# F. Calculate the cost (mean squared error) in every iteration to see how the error of the hypothesis function changes with every iteration of gradient descent.
train_mse = np.mean((y_train - lr.predict(X_train)) ** 2)
test_mse = np.mean((y_test - lr.predict(X_test)) ** 2)
print("Training Mean Squared Error:", train_mse)
print("Testing Mean Squared Error:", test_mse)

# G. Plot the cost against the number of iterations.
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

accuracy = lr.score(X_test, y_test)
print("Accuracy:", accuracy)





