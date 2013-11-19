#! /usr/bin/python2
# vim: set fileencoding=utf-8
import gzip
import cPickle
import scipy.io as sio
import numpy as np

if __name__ == '__main__':
    data = sio.loadmat('small')
    with gzip.open('flickr.pkl.gz', 'wb') as f:
        cPickle.dump([(np.asarray(data['train_set'].todense(), dtype=np.float32),
                       np.ravel(np.asarray(data['train_label']))),
                      (np.asarray(data['val_set'].todense(), dtype=np.float32),
                       np.ravel(np.asarray(data['val_label']))),
                      (np.asarray(data['test_set'].todense(), dtype=np.float32),
                       np.ravel(np.asarray(data['test_label'])))], f, -1)
