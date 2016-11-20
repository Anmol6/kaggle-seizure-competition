import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import sys
import argparse
from utils import EEGDataLoader
import pdb


# 0.00006 works fine for batch_size 64 and n_steps 512
learning_rate = 0.000009
training_iters = 100000
batch_size = 256 
display_step = 8
checkpoint_step = 512 
cross_val_step = 64 

n_channels = 16
n_hidden = 256 
n_classes = 2

n_steps = 256 

x = tf.placeholder("float", [batch_size, n_steps, n_channels])
y = tf.placeholder("float", [batch_size, n_classes])

weights = {
    'out': tf.Variable( tf.random_normal([2*n_hidden, n_classes]) )  
}

biases = {
    'out': tf.Variable( tf.random_normal([n_classes]) )
}

def BiRNN(x, weights, biases):
    # shape of x will be (batch_size, n_steps, n_channels)
    # need to transform to (n_steps, batch_size, n_channels) to comply with tf bi_rnn
    x = tf.transpose(x, [1, 0, 2])
    x = tf.unpack(x, axis = 0)

    # Forward direction cell
    #with tf.variable_scope('forward'):
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    #with tf.variable_scope('backward'):
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
    #except Exception:
    #    outputs = rnn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32, time_major = True)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Evaluate Model
# argmax returns index of largest value, along given dimension
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type = str, default=None)
    parser.add_argument('--init_from', type = str, default=None)
    parser.add_argument('--results', type = str, default=None)
    parser.add_argument('--trainf', type = str, default=None)
    parser.add_argument('--testf', type = str, default=None)
    parser.add_argument('--save', type = str, default=None)
    args = parser.parse_args()
    saver = tf.train.Saver()
    if args.sample:
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        saver.restore(sess, ckpt.model_checkpoint_path)
        # load the data for the sequence to be sampled
        # iterate over all the possible windows and calculate the average accuracy
        sample_sequence = np.load(args.sample)['data'][()]
        n = sample_sequence.shape[0]
        # total sequence length - window size
        num_windows = n-n_steps
        # iterate over the windows and get classifications
        all_pred = np.zeros((num_windows/100+1, 2), dtype=float)
        for i in xrange(0,num_windows,100):
            batch_x = [sample_sequence[i:i+n_steps]]
            cur_pred = sess.run(pred, feed_dict = {x: batch_x})[0]
            if(i % 1000 == 0):
                print(str(cur_pred[0]) + ' ' + str(cur_pred[1]))
            all_pred[i/100] = cur_pred 
        np.save(args.results, all_pred)
        
    else:
        if(args.init_from):
            ckpt = tf.train.get_checkpoint_state(args.init_from)
            saver.restore(sess, ckpt.model_checkpoint_path)
        stride = n_steps
        trainf = args.trainf 
        testf = args.testf
        savef = args.save
        print("Loading data files")
        dataloader = EEGDataLoader(trainf, testf, batch_size, n_steps, stride)  
        sess.run(init)
        step = 1
        ma_acc = 0
        ma_loss = 0
        while step < training_iters:
            try:
                batch_x, batch_y = dataloader.next()
            except StopIteration:
                print('~~~~~~~~~~~~~~~~~~~~~')
                print('~~~~~~~~EPOCH~~~~~~~~')
                print('~~~~~~~~~~~~~~~~~~~~~')
                dataloader.next_epoch()
                batch_x, batch_y = dataloader.next()
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                ma_acc = acc*0.25 + ma_acc*0.75

                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                ma_loss = loss*0.25 + ma_loss*0.75

                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc) + ", MA Loss= " + \
                      "{:.6f}".format(ma_loss) + ", MA Training= " + \
                      "{:.5f}".format(ma_acc))
            
            if step % cross_val_step == 0:
                total_acc = 0
                test_steps = 40
                for i in range(test_steps):
                    test_data, test_label = dataloader.next_test_batch()
                    cross_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_label})
                    total_acc += cross_acc
                
                total_acc = total_acc / test_steps
                print("Cross Validation Acc= " + "{:.5f}".format(total_acc))

            if step % checkpoint_step == 0:
                # Save a checkpoint
                saver.save(sess, savef + 'model.ckpt')

            step += 1
        print("Optimization Finished!")
