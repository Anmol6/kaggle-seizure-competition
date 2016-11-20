from utils import EEGDataLoader
import sys
import pdb

n_steps = 2048 
batch_size = 16 
stride = n_steps
trainf = sys.argv[1]
testf = sys.argv[2]
dataloader = EEGDataLoader(trainf, testf, batch_size, n_steps, stride)
(x,y) = dataloader.next()

pdb.set_trace()
