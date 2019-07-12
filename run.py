
import tools
import tensorflow as tf
import numpy as np
from math import floor
from time import time
from datetime import datetime
import sys
import os

def batch_normalize(Z):
    A = tf.Variable(1,name='A',dtype=tf.float32)
    B = tf.Variable(0,name='B',dtype=tf.float32)
    mu = tf.reduce_mean(Z)
    std = tf.math.reduce_std(Z)
    Z_norm = (Z - mu)/std
    Z_bn = (Z_norm * A) + B
    return Z_bn

def nn_dense_layer(X, n_neurons, name, activation=None, tied_weights=None, **options):
    if not options['TIE_WEIGHTS']:
        tied_weights=None
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.random.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        if tied_weights is not None:
            W = tf.transpose(tied_weights, name='W')
        else:
            W = tf.Variable(init, name='W')
        b = tf.Variable(tf.zeros([n_neurons]), name='b')
        Z = tf.matmul(X,W) + b
        if activation is not None:
            if options['BATCH_NORM']:
                Z_act = activation(Z)
                Z_act_bn = batch_normalize(Z_act)
                return Z_act_bn , W
            else:
                Z_act = activation(Z)
                return Z_act, W
        else:
            return Z , W

t1 = time()

##### Run Options #####

TRAIN = True
digits = [0,5]
noise_mag = 1

##### Variable Hyperparameters #####

max_n_epochs = 150
patience = 5
batch_size = 50

l1_loss_lambda = 0
l2_loss_lambda = 0.00001
TIE_WEIGHTS = True
BATCH_NORM = True

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_hidden3 = 50
n_hidden4 = 100
n_hidden5 = 300
n_outputs = 28*28

##########################################

if not TRAIN:
    run_dir = sys.argv[1]
root_modeldir = './models'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
modeldir = '{}/run_{}'.format(root_modeldir, now)
if not os.path.exists(modeldir):
    os.makedirs(modeldir)

# tools.load_data_from_csv(digits) # Original function to read in the MNIST digits from CSV and save to .npy
train_target_all = np.load('train_target.npy')
train_data_all = np.load('train_data.npy')

shuffle_index = np.random.permutation(train_data_all.shape[0])
train_target_all, train_data_all = train_target_all[shuffle_index], train_data_all[shuffle_index]
train_index = floor(train_data_all.shape[0]*(90/100))

train_data, train_target = train_data_all[:train_index], train_target_all[:train_index]
val_data, val_target = train_data_all[train_index:], train_target_all[train_index:]

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name = 'X')
noise_mag_ph = tf.placeholder(tf.float32, shape=())

X_noisy = X + noise_mag_ph*tf.random.normal(tf.shape(X))

with tf.name_scope('Autoencoder'):
    hidden_layer1, W1 = nn_dense_layer(X_noisy, n_hidden1, name='hidden_layer1', activation=tf.nn.relu, TIE_WEIGHTS=TIE_WEIGHTS, BATCH_NORM=BATCH_NORM)
    hidden_layer2, W2 = nn_dense_layer(hidden_layer1, n_hidden2, name='hidden_layer2', activation=tf.nn.relu, TIE_WEIGHTS=TIE_WEIGHTS, BATCH_NORM=BATCH_NORM)
    hidden_layer3, W3 = nn_dense_layer(hidden_layer2, n_hidden3, name='hidden_layer3', activation=tf.nn.relu, TIE_WEIGHTS=TIE_WEIGHTS, BATCH_NORM=BATCH_NORM)
    hidden_layer4, _ = nn_dense_layer(hidden_layer3, n_hidden4, name='hidden_layer4',activation=tf.nn.relu, tied_weights=W3,TIE_WEIGHTS=TIE_WEIGHTS, BATCH_NORM=BATCH_NORM)
    hidden_layer5, _ = nn_dense_layer(hidden_layer4, n_hidden5, name='hidden_layer5',activation=tf.nn.relu, tied_weights=W2,TIE_WEIGHTS=TIE_WEIGHTS, BATCH_NORM=BATCH_NORM)
    logits, _ = nn_dense_layer(hidden_layer5, n_outputs, name='logits' ,tied_weights=W1,TIE_WEIGHTS=TIE_WEIGHTS, BATCH_NORM=BATCH_NORM)

with tf.name_scope('loss'):
    l1_loss = tf.reduce_sum(W1) + tf.reduce_sum(W2) + tf.reduce_sum(W3)
    l2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
    reconstruction_loss = tf.reduce_mean(tf.square(X-logits), name='reconstruction_loss')
    total_loss = reconstruction_loss + (l2_loss_lambda*l2_loss) + (l1_loss_lambda*l1_loss)

with tf.name_scope('GradDecent'):
    optimiser = tf.train.AdamOptimizer()
    training_step = optimiser.minimize(total_loss)

initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    if TRAIN:
        initializer.run()
        lowest_val_loss = float('inf')
        steps_since_lowest = 0
        for epoch in range(max_n_epochs):
            for counter in range(len(train_target) // batch_size):
                X_batch = tools.next_batch(train_data, batch_size, counter)
                sess.run(training_step, feed_dict={X: X_batch, noise_mag_ph: noise_mag})
            train_loss = reconstruction_loss.eval(feed_dict={X: train_data, noise_mag_ph: noise_mag})
            val_loss = reconstruction_loss.eval(feed_dict={X: val_data, noise_mag_ph: noise_mag})
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                steps_since_lowest = 0
                saver.save(sess,modeldir+'/model.ckpt')
            else:
                steps_since_lowest += 1
                print('Steps since lowest: ',steps_since_lowest)
            print(epoch+1, "Train loss: ", train_loss, "Val loss: ", val_loss)
            if steps_since_lowest >= patience:
                break
    else:
        saver.restore(sess, '{}model.ckpt'.format(run_dir))
    saver.restore(sess,modeldir+'/model.ckpt')
    # encodings = hidden_layer3.eval(feed_dict={X: val_data})
    # tools.visualise_encodings(encodings,val_target) # Provides a visualisation of the middle layer embeddings
    val_data_noise = X_noisy.eval(feed_dict={X: val_data, noise_mag_ph: noise_mag})
    recons = logits.eval(feed_dict={X: val_data, noise_mag_ph: noise_mag})
    tools.visualise_recons(val_data,val_data_noise,recons)

t2 = time()
t_tot = t2-t1
print('Time elapsed: ', '{:.2f}'.format(t_tot), ' secs')
