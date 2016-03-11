import cPickle, gzip, numpy
from sklearn.datasets.base import Bunch

# Load the dataset
with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)
    
mnist = Bunch(name="MNIST dataset",
              data=np.vstack([train_set[0], valid_set[0], test_set[0]]),
              target=np.hstack([train_set[1], valid_set[1], test_set[1]]))
