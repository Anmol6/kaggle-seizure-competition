import os
import sys
import numpy as np
import re
import math
import random
import pdb
from pprint import pprint

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

'''
The strategy of the data loader is to load all the 
'''

# Traverse through whole directory and compile all as one numpy array
class EEGDataLoader:
    
    MAX_LOAD_FILES = 500 

    def __init__(self, traindir, testdir, mini_batch_size, window_size, stride):
        # Save paramaters
        self.mini_batch_size = mini_batch_size
        self.window_size = window_size
        self.stride = stride
        
        self.filenames = {0:[], 1:[]}
        # scan files and add to filenames
        if(os.path.isdir(traindir)):
            for filename in os.listdir(traindir):
                base, ext = os.path.splitext(os.path.basename(filename)) 
                if( filename.endswith('.npy') ):
                    _, _, y = re.findall('(\d+)\_(\d+)\_(\d+)', base)[0]
                    if (y == '0'):
                        self.filenames[0].append(os.path.join(traindir, filename))
                        #print( 'Found:' + base)

                    elif (y == '1'):
                        self.filenames[1].append(os.path.join(traindir, filename))
                        #print( 'Found:' + base)

        # partition filenames into batches for first epoch
        self.batched_filenames = [] 
        # setup data lists for first batch 
        self.train_array = ([],[])
        self.test_array = []
        self.cur_batch = 0
        # store indices for all data slices so we can shuffle
        self.window_indices = ([],[])
        #self.load_batch()
        self.next_epoch()
        pprint(str(self.batched_filenames))

        # stores how far we are along the examples of train_array
        self.mini_batch_index = [0,0]
        
        self.load_test_folder(testdir)

    def __iter__(self):
        return self

    def next(self):
        return self.next_mini_batch()

    def load_batch(self):
        '''
        Loads the files of the current batch cycle.
        Sets up the indices for shuffling minibatches.
        '''
        self.train_array = ([],[])
        self.window_indices = ([],[])
        self.mini_batch_index = [0,0]
        # iterate over files, load their data 
        # add each possible window to indices for minibatch loading 
        for i, filename in enumerate(self.batched_filenames[self.cur_batch][0]):
            self.train_array[0].append(np.load(filename)['data'][()])
            # get size and calculate how many windows fit in this sequence
            n = self.train_array[0][-1].shape[0]
            windows = (n-1-self.window_size)/self.stride
            # add all windows to the index
            self.window_indices[0].extend( [ (i,x) for x in xrange(-1, windows+1) ] )
        
        for i, filename in enumerate(self.batched_filenames[self.cur_batch][1]):
            self.train_array[1].append(np.load(filename)['data'][()])
            n = self.train_array[1][-1].shape[0]
            windows = (n-1-self.window_size)/self.stride
            self.window_indices[1].extend( [ (i,x) for x in xrange(-1, windows+1) ] )
        # shuffle the indices
        random.shuffle(self.window_indices[0])
        random.shuffle(self.window_indices[1])

    def next_epoch(self):
        '''
        Called after an epoch has been completed. 
        Resets data indices and repartitions filenames for batches.
        '''
        # TODO: Zero out indices and clean up stuff
        self.cur_batch = 0
        self.mini_batch_index = (0,0)
        self.train_array = ([],[])

        # scramble filenames and repartition
        random.shuffle(self.filenames[0])
        random.shuffle(self.filenames[1])

        # compute partition for the files to be loaded 
        n_pos = len(self.filenames[1])
        n_neg = len(self.filenames[0])        
        class_ratio = n_pos/float(n_neg)
        num_batches = (n_pos + n_neg) / self.MAX_LOAD_FILES
        num1 = int(round(class_ratio * self.MAX_LOAD_FILES))
        batched_0s = list(chunks(self.filenames[0], self.MAX_LOAD_FILES-num1))
        batched_1s = list(chunks(self.filenames[1], num1))
        self.batched_filenames = zip(batched_0s, batched_1s)
        self.load_batch()

    def load_test_folder(self, testf):
        self.test_array = ([],[])
        print(testf)
        if(os.path.isdir(testf)):
            for filename in os.listdir(testf):
                print(filename)
                base, ext = os.path.splitext(os.path.basename(filename)) 
                if( filename.endswith('.npy') ):
                    _, _, y = re.findall('(\d+)\_(\d+)\_(\d+)', base)[0]
                    if (y == '0'):
                        self.test_array[0].append(np.load(os.path.join(testf, filename))['data'][()])
                        #print( 'Loaded:' + base)
                    elif (y == '1'):
                        self.test_array[1].append(np.load(os.path.join(testf, filename))['data'][()])
                        #print( 'Loaded:' + base)
        
    def next_mini_batch(self):
        # generate random mini batch, make sure to keep proportion between 
        # will process all files for one epoch and generate all mini batches
        # will return when epoch is complete
        
        # positive and negative samples
        n = self.mini_batch_size/2
        mini_batch_x = np.zeros((self.mini_batch_size, self.window_size, 16))
        
        # check if we are done with this batch
        if ( self.mini_batch_index[0] + n >= len(self.window_indices[0]) ): 
            # if we are done check if we are done with the whole epoch
            # check if next batch is outside of index range
            if ( self.cur_batch + 1 >= len(self.batched_filenames) ): 
                raise StopIteration()
            else:
                print('Loading next batch')
                self.cur_batch += 1
                self.load_batch()
            
        for i in xrange(n):
            #print(str(self.mini_batch_index[0]) + ' / ' + str(len(self.window_indices[0])))
            #print()
            seq, win_t = self.window_indices[0][self.mini_batch_index[0]]
            t_steps = self.train_array[0][seq].shape[0]
            if (win_t == -1):
                mini_batch_x[i] = self.train_array[0][seq][t_steps-self.window_size:t_steps]
            else:
                start = win_t * self.stride
                mini_batch_x[i] = self.train_array[0][seq][start:start+self.window_size]
            self.mini_batch_index[0] += 1
        
        # To maintain fifty fifty ratio, this will hit the end sooner
        # We need to oversample, so shuffle indices and reset iteration
        if ( self.mini_batch_index[1] + n >= len(self.window_indices[1]) ): 
            # Shuffle and reset
            random.shuffle(self.window_indices[1])
            self.mini_batch_index[1] = 0

        for i in xrange(n):
            seq, win_t = self.window_indices[1][self.mini_batch_index[1]]
            t_steps = self.train_array[1][seq].shape[0]
            if (win_t == -1):
                mini_batch_x[n+i] = self.train_array[1][seq][t_steps-self.window_size:t_steps]
            else:
                start = win_t * self.stride
                mini_batch_x[n+i] = self.train_array[1][seq][start:start+self.window_size]
            self.mini_batch_index[1] += 1         

        mini_batch_y = np.zeros((self.mini_batch_size, 2), dtype=float)
        mini_batch_y[0:n, 0] = 1
        mini_batch_y[n:2*n, 1] = 1

        # Trust me will shuffle the same way
        rng_state = np.random.get_state()
        np.random.shuffle(mini_batch_x)
        np.random.set_state(rng_state)
        np.random.shuffle(mini_batch_y)
        
        return mini_batch_x, mini_batch_y

    def next_test_batch(self):
        # just output a random sample from the test set
        n = self.mini_batch_size/2
        batch_x = np.zeros((self.mini_batch_size, self.window_size, 16))
        indices_0 = np.random.randint(len(self.test_array[0]), size = n)
        indices_1 = np.random.randint(len(self.test_array[1]), size = n)
        
        for i in xrange(n):
            t_steps = self.test_array[0][indices_0[i]].shape[0]
            index = np.random.randint(-1, (t_steps-1-self.window_size)/self.stride)
            if (index == -1):
                batch_x[i] = self.test_array[0][indices_0[i]][t_steps-self.window_size:t_steps]
            else:
                batch_x[i] = self.test_array[0][indices_0[i]][index*self.stride:index*self.stride+self.window_size]
            
        for i in xrange(n):
            t_steps = self.test_array[1][indices_1[i]].shape[0]
            index = np.random.randint(-1, (t_steps-1-self.window_size)/self.stride)
            if (index == -1):
                batch_x[n+i] = self.test_array[1][indices_1[i]][t_steps-self.window_size:t_steps]
            else:
                batch_x[n+i] = self.test_array[1][indices_1[i]][index*self.stride:index*self.stride+self.window_size]
        
        batch_y = np.zeros((self.mini_batch_size, 2), dtype=float)
        batch_y[0:n, 0] = 1
        batch_y[n:2*n, 1] = 1

        # Trust me will shuffle the same way
        rng_state = np.random.get_state()
        np.random.shuffle(batch_x)
        np.random.set_state(rng_state)
        np.random.shuffle(batch_y)
        
        return batch_x, batch_y
