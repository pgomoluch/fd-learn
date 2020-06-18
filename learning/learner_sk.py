import numpy as np
import os

import plan_reader

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

features_train, features_test, labels_train, labels_test = plan_reader.read()

############################
#### FOR FINAL TRAINING ####
############################
#features_train = np.append(features_train, features_test, 0)
#labels_train = np.append(labels_train, labels_test, 0)
############################

print "Learning from {} examples.".format(len(labels_train))
print "{} features.".format(len(features_train[0]))
print "Mean label: {}".format(np.mean(labels_train))

#reg = LinearRegression()
reg = Ridge()

reg.fit(features_train, labels_train)

print 'Model:'
print reg.intercept_
print reg.coef_

model_file = open('model.txt', 'w')
model_file.write('{}\n\n'.format(reg.intercept_))
for c in reg.coef_:
    model_file.write('{}\n'.format(c))
model_file.close()

labels_pred = reg.predict(features_train)
err_train = mean_squared_error(labels_train, labels_pred)

labels_pred2 = reg.predict(features_test)
err_test = mean_squared_error(labels_test, labels_pred2)

print 'Train MSE:', err_train
print 'Test MSE:', err_test
