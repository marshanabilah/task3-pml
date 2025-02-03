
import numpy as np
import sklearn
import sklearn.datasets
import scipy

def get_true_beta(p, scale = 1.0):
    prespecifiedBetaPart = np.asarray([1, 1.5, -2, 2.5]) * scale
    beta = np.zeros(p)
    beta[0:prespecifiedBetaPart.shape[0]] = prespecifiedBetaPart
    return beta

# some linear data without intercept
def linear_regression_data(n, beta, rho = 0.0,  sigma=1.0):
    p = beta.shape[0]
    assert(p >= 4)

    covariance_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            covariance_matrix[i, j] = rho**(abs(i - j))

    X = np.random.multivariate_normal(np.zeros(p), covariance_matrix, n)
    epsilon = np.random.randn(n)

    y = np.dot(X, beta) + sigma * epsilon
    gamma = np.where(beta != 0, 1, 0) # indicates which position of beta is non-zero
    return X, y, gamma


# same data as in "Regression Shrinkage and Selection via the Lasso"
def lasso_linear(n = 20, p = 8, rho = 0.5, sigma=3.0):
    assert(p >= 8)
    prespecifiedBetaPart = np.array([3, 1.5, 0, 0, 2, 0, 0, 0])
    beta = np.zeros(p)
    beta[0:prespecifiedBetaPart.shape[0]] = prespecifiedBetaPart

    data_dim = beta.shape[0]
    assert(data_dim == p)
    covariance_matrix = np.zeros((data_dim, data_dim))
    for i in range(data_dim):
        for j in range(data_dim):
            covariance_matrix[i, j] = rho**(abs(i - j))

    X = np.random.multivariate_normal(np.zeros(data_dim), covariance_matrix, n)
    epsilon = np.random.randn(n)

    y = np.dot(X, beta) + sigma * epsilon
    gamma = np.where(beta != 0, 1, 0)
    return X, y, beta, gamma


def generateLogRegData(beta, correlationMatrix, intercept, n):
    
    meanVec = np.zeros(beta.shape[0])
    X = np.random.multivariate_normal(meanVec, correlationMatrix, n)
    logOdds = np.matmul(X, beta) + intercept
    
    occurrenceProbabilities = 1.0 / (1.0 + np.exp(-logOdds))
     
    # print("occurrenceProbabilities = ")
    # with numpy.printoptions(precision=3):
    #    print(occurrenceProbabilities)
    
    y = scipy.stats.bernoulli.rvs(occurrenceProbabilities)
    
    return X, y

# generate data as in "Contraction properties of shrinkage priors in logistic regression"
# increase intrinisic (aleotory uncertainty) by lowering "scale"
def generateLinearLogisticRegressionData(n, d, scale = 1.0, rho = 0.2):
    
    intercept = 0.0
     
    prespecifiedBetaPart = np.asarray([1, 1.5, -2, 2.5]) * scale
    beta = np.zeros(d)
    beta[0:prespecifiedBetaPart.shape[0]] = prespecifiedBetaPart
        
    
    correlationMatrix = generateCorrelationMatrix(d, rho)
        
    return generateLogRegData(beta, correlationMatrix, intercept, n)


def generateCorrelationMatrix(d, rho):

    correlationMatrix = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            correlationMatrix[i,j] = rho ** np.abs(i-j)
        
    return correlationMatrix

def generateNonLinearLogisticRegressionData(n, d, rho = 0.2):
    assert(d >= 2)
    
    X_2dPart, y = sklearn.datasets.make_gaussian_quantiles(cov=1.0,
                                     n_samples=n, n_features=2,
                                     n_classes=2, random_state=44231)
    
    
    correlationMatrix = generateCorrelationMatrix(d-2, rho)
    meanVec = np.zeros(d-2)
    X_remaining = np.random.multivariate_normal(meanVec, correlationMatrix, n)
    
    X = np.hstack((X_2dPart, X_remaining))
    
    return X, y