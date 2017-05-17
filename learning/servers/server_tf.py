import os
import socket
import struct

import numpy as np
import tensorflow as tf

#import plan_reader

socket_path = 'fd-learn-socket'
double_size = 8
FEATURE_NUMBER = 5


def evaluate(model, state):
    state = np.array([state])
    [value] = model.predict({'x':state}, as_iterable=False)
    return value


feature_list = [tf.contrib.layers.real_valued_column("x", dimension=5)]
hidden_units=[5,5]
model = tf.contrib.learn.DNNRegressor(feature_columns=feature_list, hidden_units=hidden_units, model_dir='../tf-model')

# DEBUG
#features_train, features_test, labels_train, labels_test = plan_reader.read()
#pred = list(model.predict({"x":features_test[[10,11,12]]}, as_iterable=True))
#print pred

try:
    os.unlink(socket_path)
except:
    pass

ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
ls.bind(socket_path)
ls.listen(5)

conn, _ = ls.accept()

closed = False
while not closed:
    data = ''
    while len(data) < FEATURE_NUMBER * double_size:
        data_part = conn.recv(4096)
        if not data_part:
            print 'Connection closed by the client'
            closed = True
            break
        data += data_part
    
    if not closed:
        features = struct.unpack('d'*FEATURE_NUMBER, data)
        result = evaluate(model, features)
        conn.send(struct.pack('d',result))

conn.close()
ls.close()
os.unlink(socket_path)

