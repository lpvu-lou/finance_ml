import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pytest
from finance_ml.linear_models import LinearRegression

def test_1():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 3, 6])

    model = LinearRegression(use_intercept=False)
    model.fit(X, y)
    preds = model.predict(X)

    print("Intercepts:", model.intercept)
    print("Predictions:", preds)
    print("Expected:", y)

def test_2():
    X = np.array([[0], [1], [2]])
    y = np.array([1, 3, 5]) 

    model = LinearRegression(use_intercept=True)
    model.fit(X, y)
    preds = model.predict(X)

    print("Intercept:", model.intercept)
    print("Predictions:", preds)
    print("Expected:", y)

if __name__ == "__main__":
    test_1()
    test_2()





    