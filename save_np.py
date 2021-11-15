import numpy as np
import sys

if __name__ == "__main__":
    """
    Covert glove.txt file into two parts
    glove_vocab.txt stores words
    glove_vector.npy stores word vectors
    EX: python save_np.py ~/data glove.840B.300d.txt
    """
    root = '/Users/lidanyang/data'
    filename = 'glove.840B.300d.txt'
    v_list = []
    pwd_origin = str(root) + '/' + str(filename)
    pwd_vocab = str(root) + '/glove_vocab.txt'
    f1 = open(pwd_origin, "r", encoding='utf-8')
    f2 = open(pwd_vocab, "w", encoding='utf-8')
    for line in f1.readlines():
        f2.write(line.split(' ')[0] + '\n')
        v = line.split(' ')[1:]
        v_list.append(np.array(v, dtype=np.float32))
    matrix = np.array(v_list)
    np.save(str(root) + '/glove_vector.npy', matrix)
    f1.close()
    f2.close()
