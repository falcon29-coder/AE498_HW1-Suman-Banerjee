"""This module contains functions to perform multiple linear regression for HW 1 in AE 498 Computational Systems Engineering.

"""

import numpy as np
import scipy


def model_fit(X, y):
    """Function to fit a multiple linear regression model to a given dataset using ordinary least squares estimation.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. The first column should be all ones.
    y : array_like
        Output/response data (n elements), where each element is an observation.

    Returns
    -------
    coefficients : array_like
        Estimated model coefficients (1 + p elements).
    y_predicted : array_like
        Predicted responses for input data (n elements).

    """

    # YOUR CODE HERE- Suman's code
    X=np.asarray(X)
    y=np.asarray(y)
    coefficients=np.linalg.solve(X.T@X,X.T@y)
    y_predicted=X@coefficients
    return coefficients, y_predicted



def anova(y, y_predicted, p):
    """Function to perform an ANOVA F-test for a multiple linear regression model.

    Parameters
    ----------
    y : array_like
        Output/response data (n elements), where each element is an observation.
    y_predicted : array_like
        Predicted responses for input data (n elements).
    p : int
        Number of predictors.

    Returns
    -------
    p_value : float
        P-value for model F-statistic.
    f_statistic : float
        F-statistic for model F-test.

    """

    # YOUR CODE HERE-Suman's code
    y=np.asarray(y)
    y_predicted=np.asarray(y_predicted)
    n=len(y)
    y_mean=np.mean(y)
    ESS=np.sum((y_predicted-y_mean)**2) #Error Sum of squares
    RSS=np.sum((y_predicted-y)**2) #Residual Sum of squares
    ESS_p=ESS/p
    RSS_n_p_1=RSS/(n-p-1)
    f_statistic=ESS_p/RSS_n_p_1
    p_value = scipy.stats.f.sf(f_statistic,p,n-p-1)
    return p_value, f_statistic


def coefficient_tests(X, coefficients, y, y_predicted):
    """Function to perform hypothesis t-tests for coefficients of a multiple linear regression model.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. The first column should be all ones.
    coefficients : array_like
        Estimated model coefficients (1 + p elements).
    y : array_like
        Output/response data (n elements), where each element is an observation.
    y_predicted : array_like
        Predicted responses for input data (n elements).

    Returns
    -------
    p_values : list_like
        P-values for coefficient t-statistics.
    t_statistics : list_like
        T-statistics for coefficient t-tests.

    """

    # YOUR CODE HERE-Suman's code
    X=np.asarray(X)
    coefficients=np.asarray(coefficients)
    y=np.asarray(y)
    y_predicted=np.asarray(y_predicted)
    n=len(y)
    p=X.shape[1]-1 # (length of column of X)-1=p+1-1=p
    RSS=np.sum((y_predicted-y)**2) # Residual Sum of squares
    RSE=RSS/(n-p-1) # Residual standard error
    covariance_matrix=RSE * np.linalg.inv(X.T @ X)
    standard_errors=np.sqrt(np.diag(covariance_matrix))
    t_statistics=coefficients/standard_errors
    p_values=2 * scipy.stats.t.sf(np.abs(t_statistics),n-p-1) # DOF=n-p-1
    return p_values, t_statistics




def normalize_data(X):
    """Function to normalize input data to [-1, 1].

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. The first column should be all ones.

    Returns
    -------
    X_normalized : array_like
        Input data (n x (p + 1) dimensions) normalized to the range [-1, 1].

    """

    n, n_columns = np.shape(X)

    # normalize each column
    X_normalized = np.ones((n, n_columns))
    for column in range(n_columns):
        x_column = X[:, column]
        x_max = np.max(x_column)
        x_min = np.min(x_column)
        if x_max != x_min:
            x_column_normalized = [
                (x - (x_max + x_min) / 2) / ((x_max - x_min) / 2) for x in x_column
            ]
            X_normalized[:, column] = x_column_normalized

    return X_normalized
