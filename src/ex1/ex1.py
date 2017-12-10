import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('ex1data1.txt', delimiter=',')
X = data[:, 0:1] # using slice indexing (0:1 instead of 0) maintains rank-2 array
y = data[:, 1:2] # using slice indexing (1:2 instead of 1) maintains rank-2 array
m = len(y) # number of training examples

# Plot Data
#print 'Plotting Data ...'
#plt.plot(X, y, 'rx', markersize=10);            # Plot the data
#plt.ylabel('Profit in $10,000s');               # Set the y-axis label
#plt.xlabel('Population of City in 10,000s');    # Set the x-axis label
#plt.show()

X = np.concatenate((np.ones((m, 1)), X), axis=1) # Add a column of ones to X
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

def computeCost(X, y, theta):
    h = np.dot(X, theta) # (97,2) * (2,1) = (97,1)
    err = h - y # (97,1) - (97,1) = (97,1)
    return 1.0 / (2.0 * m) * np.dot(err.T, err) # (2,97) * (97,1) = (2,1)

print 'Testing the cost function ...'
# compute and display initial cost
J = computeCost(X, y, theta)
print 'With theta = [0 ; 0]\nCost computed = ' + str(J)
print 'Expected cost value (approx) 32.07'

# further testing of the cost function
J = computeCost(X, y, np.array([[-1], [2]]));
print '\nWith theta = [-1 ; 2]\nCost computed = ' + str(J)
print 'Expected cost value (approx) 54.24\n'

def gradientDescent(X, y, theta, alpha, num_iters):
    for i in range(0, num_iters):
        h = np.dot(X, theta) # (97,2) * (2,1) = (97,1)
        err = h - y # (97,1) - (97,1) = (97,1)
        theta_change = alpha / m * np.dot(X.T, err) # (2,97) * (97,1) = (2,1)
        theta = theta - theta_change # (2,1) - (2,1) = (2,1)
    return theta

print '\nRunning Gradient Descent ...\n'
# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print 'Theta found by gradient descent:'
print theta
print 'Expected theta values (approx)'
print ' -3.6303\n  1.1664\n\n'

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print 'For population = 35,000, we predict a profit of ' + str(predict1*10000)
predict2 = np.dot([1, 7], theta)
print 'For population = 70,000, we predict a profit of ' + str(predict2*10000);
