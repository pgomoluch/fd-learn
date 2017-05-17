import plan_reader

columns = [0,1,2,3,4,5,6] # no FF or CEA derived features

def write_matrix(matrix, path):
    out = open(path, 'w')
    for row in matrix:
        for i in row:
            out.write('{} '.format(i))
        out.write('\n')
    out.close()

def write_vector(vector, path):
    out = open(path, 'w')
    for i in vector:
        out.write('{}\n'.format(i))
            

features_train, features_test, labels_train, labels_test = plan_reader.read()

write_matrix(features_train, 'features_train.txt')
write_matrix(features_test, 'features_test.txt')
write_vector(labels_train, 'labels_train.txt')
write_vector(labels_test, 'labels_test.txt')

#features_full = np.append(features_train, features_test, axis=0)
#labels_full = np.append(labels_train, labels_test, axis=0)
#write_matrix(features_full, 'features_full.txt')
#write_vector(labels_full, 'labels_full.txt')
