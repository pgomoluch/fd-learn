import random
import tensorflow as tf
import numpy as np

import plan_reader

N_FEATURES = 18

features_train, features_test, labels_train, labels_test = plan_reader.read()

print "Learning from {} examples.".format(len(labels_train))
print "{} features.".format(len(features_train[0]))
print "Mean label: {}".format(np.mean(labels_train))

#feature_list = [\
#    tf.contrib.layers.real_valued_column("h_dist", dimension=1),\
#    tf.contrib.layers.real_valued_column("ff", dimension=1),\
#    tf.contrib.layers.real_valued_column("ff_op", dimension=1),\
#    tf.contrib.layers.real_valued_column("ff_ign_eff", dimension=1),\
#    tf.contrib.layers.real_valued_column("ff_avg_ign_eff", dimension=1),\
#]
feature_list = [tf.contrib.layers.real_valued_column("x", dimension=N_FEATURES)]

def input_fn_train():
    ii = np.random.choice(len(labels_train), 1000)
    return {"x": tf.constant(features_train[ii])}, tf.constant(labels_train[ii])

def input_fn_test():
    return {"x": tf.constant(features_test)}, tf.constant(labels_test)

def input_fn_test_on_train():
    return {"x": tf.constant(features_train)}, tf.constant(labels_train)

#input_fn_train = tf.contrib.learn.io.numpy_input_fn({"x":features_train}, labels_train, batch_size=1)
#input_fn_test = tf.contrib.learn.io.numpy_input_fn({"x":features_test}, labels_test, batch_size=1)

#reg = tf.contrib.learn.LinearRegressor(feature_columns=feature_list)
reg = tf.contrib.learn.DNNRegressor(feature_columns=feature_list, hidden_units=[18,9], model_dir='./tf-model')
print 'Starting training...'
reg.fit(input_fn=input_fn_train, steps=100000)
print 'Finished training.'

train_ev = reg.evaluate(input_fn=input_fn_test_on_train, steps=1)
test_ev = reg.evaluate(input_fn=input_fn_test, steps=1)
pred = list(reg.predict({"x":features_test[[10]]}, as_iterable=True))
#pred = list(reg.predict(x={"x": features_test}, as_iterable=True)) # works
#pred = reg.predict(input_fn=input_fn_test, as_iterable=False) # works
print pred

print 'Training set size:', len(labels_train)
print 'Test set size:', len(labels_test)
print 'Training evaluation:', train_ev
print 'Test evaluation:', test_ev
