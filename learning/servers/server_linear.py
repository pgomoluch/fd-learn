import os
import socket
import struct

socket_path = 'fd-learn-socket'
model_path = 'model.txt'
double_size = 8

model_file = open(model_path)
numbers = [float(x) for x in model_file.read().split()]
model_file.close()
intercept = numbers[0]
weights = numbers[1:]

ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
ls.bind(socket_path)
ls.listen(5)

conn, _ = ls.accept()

closed = False
while not closed:
    data = ''
    while len(data) < len(weights) * double_size:
        data_part = conn.recv(4096)
        if not data_part:
            print 'Connection closed by the client'
            closed = True
            break
        data += data_part
    
    if not closed:
        features = struct.unpack('d'*len(weights), data)
        #print features
        result = intercept
        for (w,x) in zip(weights, features):
            result += w*x
        conn.send(struct.pack('d',result))

conn.close()
ls.close()
os.unlink(socket_path)
