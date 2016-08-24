import numpy as np
from sklearn.linear_model import Lasso

#read data from file into an numpy array
data = np.array([[content for content in map(float, line.strip("\n").split(","))]\
    for line in open("../datasets/lasso_100c_1kr.csv", 'r').readlines()])

#slice features and labels 
x = data[:, 0:-1]
y = data[:,-1]

#initiallize lasso with alpha and max_iter
clf = Lasso(alpha=0.01, max_iter=100)

#train lasso
clf.fit(x, y)

#get the weights and intercept of trained model
print "coef:{0}\nintercept:{1}".format(clf.coef_, clf.intercept_)

#get the r-squared error agains actual and predicted labels
print clf.score(x, y)
