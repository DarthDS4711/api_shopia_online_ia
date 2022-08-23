import numpy as np


class LinnearRegression:
    def __init__(self):
        self.__n_iterations = 6000
        self.__alpha = 0.000003

    def __compute_error(self, original_y, predicted_y):
        N = original_y.shape[0]
        error = np.sum((original_y - predicted_y) ** 2) / N
        return error
    
    def compute_model(self, w, b, x):
        return w * x + b
    
    def __downward_gradient(self, w_:np.float64, b_:np.float64, x, y):
        N = x.shape[0]
        dw = -(2 / N) * np.sum(x * (y - (w_ * x + b_)))
        db = -(2 / N) * np.sum(y - (w_ * x + b_))
        w_ = np.double(w_ - self.__alpha * dw)
        b_ = np.double(b_ - self.__alpha * db)
        return w_, b_
    
    def train_linnear_regression(self, x, y):
        try:
            np.random.seed(2)
            w = np.random.random(1)
            b = np.random.random(1)
            error = np.zeros((self.__n_iterations, 1), dtype=np.float64)
            for iteration in range(self.__n_iterations):
                [w, b] = self.__downward_gradient(w, b, x, y)
                predicted_y = self.compute_model(w, b, x)
                error[iteration] = self.__compute_error(y, predicted_y)
            return w, b
        except Exception as e:
            print(str(e))
