# Denoising Autoencoder

By Oscar Bennett, 2019

This is a [TensorFlow](https://www.tensorflow.org) implementation of a simple denoising autoencoder applied to the MNIST dataset. A denoising autoencoder is a type of encoding-decoding neural network which compresses data down to a lower dimensional representation in an unsupervised manner and can learn to remove noise in the process. A nice explanation of the theory can be found [here](http://deeplearning.net/tutorial/dA.html).

The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) is a large collection of images of hand written digits. Its a nice and simple dataset to apply data science and machine learning methods to in order to demonstrate their use and benchmark their performance.

Some example results after applying gaussian noise to MNIST digit images of 0s and 5s is shown here:

Original:
<img src=resources/orig.png width=100%>
Corrupted:
<img src=resources/corrupted.png width=100%>
Reconstruction:
<img src=resources/recon.png width=100%>

As you can see it can get pretty good at finding a signal in a lot of noise!

To run, clone the repo and then execute the following commands:
```
> conda create -n Denoise_AE python=3.7 pip
> source activate Denoise_AE
> pip install -r requirements.txt
> python run.py
```

This will setup the environment in [conda](https://docs.conda.io/en/latest/), train the model, save it, and then generate and plot some examples of reconstructions chosen randomly from the validation set. (like shown above)

The final trained tensorflow model checkpoints are saved in a `model/` directory.

To improve the performance of the model I implemented a few basic model and training features such [batch normalisation](https://arxiv.org/abs/1502.03167), [early stopping](https://en.wikipedia.org/wiki/Early_stopping), [L2 regularisation](https://medium.com/datadriveninvestor/l1-l2-regularization-7f1b4fe948f2), and [encoding-decoding weight tying](https://stackoverflow.com/questions/36889732/tied-weights-in-autoencoder). If you're curious about these techniques just follow the links to discover more.

The effect of these techniques as well as changes to the structure of the network can be explored by altering the hyperparameter variables near the top of the `run.py` file:

```python
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
```

Have fun playing with it. I've also included a `noise_mag_ph` tensorflow placeholder which allows you to experiment with injecting different amounts of noise at different stages of the training and inference process. Just alter its value in the `feed_dict`s.

Please feel free to let me know about any suggestions or issues!
