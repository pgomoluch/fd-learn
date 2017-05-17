import numpy as np
import os
import socket
import struct
import learner_sk

socket_path = 'servers/fd-learn-socket'
double_size = 8

n_features = len(learner_sk.features_train[0])

ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
ls.bind(socket_path)
ls.listen(5)

conn, _ = ls.accept()

closed = False
while not closed:
    data = ''
    while len(data) < n_features * double_size:
        data_part = conn.recv(4096)
        if not data_part:
            print 'Connection closed by the client'
            closed = True
            break
        data += data_part
    
    if not closed:
        features = struct.unpack('d'*n_features, data)
        result = learner_sk.reg.predict(np.array(features).reshape(1,-1))
        conn.send(struct.pack('d',result))

conn.close()
ls.close()
os.unlink(socket_path)
