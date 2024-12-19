import matplotlib.pyplot as plt
import numpy as np

def linearRegression(x, y):
    # Add column of ones to x for bias term
    X = np.column_stack([np.ones(len(x)), x])
    
    # Calculate weights using normal equation
    # w = (X^T X)^-1 X^T y
    weights = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
 
    # weights[0] is bias, weights[1] is slope
    bias = weights[0]
    slope = weights[1]
    
    # Calculate predictions
    y_pred = np.dot(X, weights)

    
    # Calculate error (MSE)
    mse = np.mean((y_pred - y) ** 2)
    
    # Calculate R-squared (how well our line fits)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    return slope, bias, y_pred, mse, r_squared

def main():
    np.random.seed(0)
    x = 4 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1) * 0.5
    
    # Get results
    slope, bias, y_pred, mse, r_squared = linearRegression(x, y)
    
    # Print results
    print(f"Estimated slope: ", slope)
    print(f"Estimated bias: ", bias)
    print(f"Mean Squared Error: ", mse)
    print(f"R-squared Score: ", r_squared)
    
    # Plot results
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, y_pred, color='red', label='Linear regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Results')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()