import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def load_data_from_csv(digits):
    train = pd.read_csv('../train.csv')

    train_target_all = train['label']
    train_data_all = train.drop(columns='label')

    train_target_all = train_target_all.values.astype(np.int8)
    train_data_all = train_data_all.values.astype(np.float16)
    mask_array = np.array([False]*len(train_target_all))

    for digit in digits:
        digit_mask = np.array((train_target_all == digit))
        mask_array = np.logical_or(mask_array,digit_mask)

    train_data = train_data_all[mask_array]
    train_target = train_target_all[mask_array]
    train_data /= 255
    np.save('train_target.npy',train_target)
    np.save('train_data.npy',train_data)

def print_number_image(target, image_data):
    print('Target number is: ', target)
    for i in range(28):
        print(list(image_data[28*i:(28*i + 28)]))

def next_batch(X, size, counter):
    X_batch = X[size*counter:size*(counter+1),:]
    return X_batch

def plot_digit(digit_values):
    shape = (28,28)
    digit = digit_values.reshape(shape)
    plt.imshow(digit.astype(np.float32), cmap="Greys", interpolation='nearest')
    plt.show()

def visualise_encodings(encodings, val_target):
    pca = PCA(n_components=2)
    encodings_pca = pca.fit_transform(encodings)
    colour_list = ['blue','red','orange','green']
    number_of_digits = len(np.unique(val_target))
    colours = [colour_list[i] for i in range(number_of_digits)]
    plt.scatter(encodings_pca[:,0],encodings_pca[:,1],c=colours)
    plt.show()

def visualise_recons(val_data,val_data_noise,recons):
    shape = (28,28)
    num_to_display = 8
    for i in range(num_to_display):
        i +=1
        digit_values = val_data[i,:]
        digit = digit_values.reshape(shape)
        digit_values_noise = val_data_noise[i,:]
        digit_nosie = digit_values_noise.reshape(shape)
        recon_data = recons[i,:]
        recon_image = recon_data.reshape(shape)
        plt.subplot(3,num_to_display,i)
        plt.imshow(digit.astype(np.float32), cmap="Greys", interpolation='nearest')
        plt.subplot(3,num_to_display,i+num_to_display)
        plt.imshow(digit_nosie.astype(np.float32), cmap="Greys", interpolation='nearest')
        plt.subplot(3,num_to_display,i+num_to_display+num_to_display)
        plt.imshow(recon_image.astype(np.float32), cmap="Greys", interpolation='nearest')
    plt.show()
