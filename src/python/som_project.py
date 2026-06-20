#!/usr/bin/env python3
# Auto-generated from notebooks/Proj1.ipynb and cleaned up for script usage.
from __future__ import annotations

import os
import random
import copy
import pickle
import warnings
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from minisom import MiniSom
from pprint import pprint
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUTPUT_DIR = ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

# %% [markdown]
# # importing packages
# %%
# %% [markdown]
# # Task 1: Clustering using SOM
# %% [markdown]
# **Preparing Dataset italicized text**
# %%
""""
	see the properties of digits and print its description
		to see the attributes and shape of each of which,
		uncomment the section bellow
"""
digits = load_digits()
#pprint(digits)
#print(digits.DESCR)
#print(digits.data.shape)
#print(digits.images.shape)
#print(digits.target.shape)


raw_data = scale(digits.data)
raw_label = digits.target
raw_images = digits.images;

"""
		Shuffle and Split Data to two categories, Train Data and Test Data with
		ratio of 80%
"""
combined_data = list(zip(raw_data, raw_label, raw_images))
random.shuffle(combined_data)
shuffled_data, shuffled_label, shuffled_image = zip(*combined_data)


all_data = shuffled_data
all_label = shuffled_label
all_images = shuffled_image

# %% [markdown]
# **Creating and Initializing SOM models**
# %%
som_4X4   = MiniSom(4,  4,  64, sigma=3, learning_rate=0.5, neighborhood_function='triangle')
som_20X20 = MiniSom(20, 20, 64, sigma=4, learning_rate=0.5, neighborhood_function='triangle')

# %% [markdown]
# **Training Networks**
# %%
#trainning SOM Models on All Data
som_4X4.train  (all_data, 10000, verbose=True)
som_20X20.train(all_data, 10000, verbose=True)

# %% [markdown]
# **Visualization**
# 
#   
# %% [markdown]
# 4X4 Model
# %%
plt.figure(figsize=(8, 8))
wmap_4X4 = {}
hcntr4 = np.zeros((4,4,), dtype=int);
im = 0
for x, t in zip(all_data, all_label):  # scatterplot
    w = som_4X4.winner(x)
    wmap_4X4[w] = im
    hcntr4[w] = hcntr4[w]+1
    plt. text(w[0]+0.5,  w[1]+0.5,  str(t),
              color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 22})
    im = im + 1
plt.axis([0, som_4X4.get_weights().shape[0], 0,  som_4X4.get_weights().shape[1]])
plt.savefig('outputs/figures/SOM_4X4_clustering_map.png')
plt.show()
print("")

plt.figure(figsize=(8, 8), facecolor='white')
cnt = 0
for j in reversed(range(4)):  # images mosaic
    for i in range(4):
        plt.subplot(4, 4, cnt+1, frameon=False,  xticks=[],  yticks=[])
        if (i, j) in wmap_4X4:
            plt.imshow(all_images[wmap_4X4[(i, j)]],
                       cmap='Greys', interpolation='nearest')
        else:
            plt.imshow(np.zeros((8, 8)),  cmap='Greys')
        cnt = cnt + 1

plt.tight_layout()
plt.savefig('outputs/figures/SOM_4X4_data.png')
plt.show()
print("")


#
# Hit Map
#
print("The hit map matrix for 4X4 model:\r\n(flipped horizontally): ")
print(np.flipud(hcntr4))
print("")
nhcntr4 = hcntr4 / hcntr4.max()
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
all_patches = []
cmap = cm.get_cmap('viridis')
for i in range(4):
    for j in range(4):
        edge = nhcntr4[j][i];
        rect_color = cmap(nhcntr4[j][i])
        all_patches.append(patches.Rectangle((i+0.5-edge/2, j+0.5-edge/2), edge, edge, linewidth=1, fill=1, facecolor=rect_color,))
ax.add_collection(PatchCollection(all_patches))
pc = PatchCollection(all_patches, cmap=cmap)  # Set cmap here
pc.set_array(np.transpose(nhcntr4).flatten())  # Set array for color mapping
ax.add_collection(pc)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=hcntr4.min(), vmax=hcntr4.max()))
sm.set_array(np.transpose(nhcntr4).flatten())  # Set array for color bar
plt.colorbar(sm, ax=ax)
plt.axis([-0.5, som_4X4.get_weights().shape[0]+0.5, -0.5,  som_4X4.get_weights().shape[1]+0.5])
plt.savefig('outputs/figures/SOM_4X4_Hit_map.png')
plt.show()
print("")

#
#   Dead Neurons
#
print("Dead Neurons:")
dead_cntr = 0
for i in range(4):
    for j in range(4):
        if hcntr4[i,j] == 0:
            print("\tNeuron (", i, ", ", j, ") is dead.")
            dead_cntr += 1
if dead_cntr == 0:
    print("\tThere is no dead neuron in 4X4 SOM model.")
else:
    print("\r\nThere are ", dead_cntr, " dead neurons in 4X4 SOM model.")

# %% [markdown]
# 20X20 Model
# %%
plt.figure(figsize=(8, 8))
wmap_20X20 = {}
hcntr20 = np.zeros((20,20,), dtype=int);
im = 0
for x, t in zip(all_data, all_label):  # scatterplot
    w = som_20X20.winner(x)
    wmap_20X20[w] = im
    hcntr20[w] = hcntr20[w]+1
    plt. text(w[0]+0.5,  w[1]+0.5,  str(t),
              color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 16})
    im = im + 1
plt.axis([0, som_20X20.get_weights().shape[0]+0.5, 0,  som_20X20.get_weights().shape[1]+0.5])
plt.savefig('outputs/figures/SOM_20X20_clustering_map.png')
plt.show()
print("")

plt.figure(figsize=(8, 8), facecolor='white')
cnt = 0
for j in reversed(range(20)):  # images mosaic
    for i in range(20):
        plt.subplot(20, 20, cnt+1, frameon=False,  xticks=[],  yticks=[])
        if (i, j) in wmap_20X20:
            plt.imshow(all_images[wmap_20X20[(i, j)]],
                       cmap='Greys', interpolation='nearest')
        else:
            plt.imshow(np.zeros((8, 8)),  cmap='Greys')
        cnt = cnt + 1

plt.tight_layout()
plt.savefig('outputs/figures/SOM_20X20_data.png')
plt.show()
print("")


#
# Hit Map
#
print("The hit map matrix for 20X20 model (flipped horizontally): ")
print(np.flipud(hcntr20))
print("")
nhcntr20 = hcntr20 / hcntr20.max()
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
all_patches = []
cmap = cm.get_cmap('viridis')
for i in range(20):
    for j in range(20):
        edge = nhcntr20[j][i];
        rect_color = cmap(nhcntr20[j][i])
        all_patches.append(patches.Rectangle((i+0.5-edge/2, j+0.5-edge/2), edge, edge, linewidth=1, fill=1, facecolor=rect_color,))
ax.add_collection(PatchCollection(all_patches))
pc = PatchCollection(all_patches, cmap=cmap)  # Set cmap here
pc.set_array(np.transpose(nhcntr20).flatten())  # Set array for color mapping
ax.add_collection(pc)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=hcntr20.min(), vmax=hcntr20.max()))
sm.set_array(np.transpose(nhcntr20).flatten())  # Set array for color bar
plt.colorbar(sm, ax=ax)
plt.axis([-0.5, som_20X20.get_weights().shape[0]+0.5, -0.5,  som_20X20.get_weights().shape[1]+0.5])
plt.savefig('outputs/figures/SOM_20X20_Hit_map.png')
plt.show()
print()

#
#   Dead Neurons
#
print("Dead Neurons:")
dead_cntr = 0
for i in range(20):
    for j in range(20):
        if hcntr20[i,j] == 0:
            print("\tNeuron (", i, ", ", j, ") is dead.")
            dead_cntr += 1
if dead_cntr == 0:
    print("\tThere is no dead neuron in 20X20 SOM model.")
else:
    print("\r\nThere are ", dead_cntr, " dead neurons in 20X20 SOM model.")

# %% [markdown]
# **Calculating the Quantization Error**
# %%
#
# 4X4
#
Wghts_4X4   = som_4X4.get_weights();
QE_acc = 0
for x in all_data:
    # calculate the distance between the data and each neurons and find the nearest
    sumi = np.zeros((4,4,))
    for i in range(4):
        for j in range(4):
            sumi[i,j] = np.sqrt( np.sum( np.square(Wghts_4X4[i,j] - x) ) )
    QE_acc += sumi.min();
print("The Quantization error of 4X4   neuron SOM model is: ", QE_acc/len(all_data))


#
# 20X20
#
Wghts_20X20   = som_20X20.get_weights();
QE_acc = 0
for x in all_data:
    # calculate the distance between the data and each neurons and find the nearest
    sumi = np.zeros((20,20,))
    for i in range(20):
        for j in range(20):
            sumi[i,j] = np.sqrt( np.sum( np.square(Wghts_20X20[i,j] - x) ) )
    QE_acc += sumi.min();
print("The Quantization error of 20X20 neuron SOM model is: ", QE_acc/len(all_data))

# %% [markdown]
# # Task 2: Classification of Digits Dataset Using SOM
# %% [markdown]
# **Data Preparation**
# %%
split_point = int(0.8 * len(shuffled_label))


train_data = shuffled_data[:split_point]
train_label = shuffled_label[:split_point]
train_images = shuffled_image[:split_point]

test_data = shuffled_data[split_point:]
test_label = shuffled_label[split_point:]
test_images = shuffled_image[split_point:]

# %% [markdown]
# **Retrainning SOM Models on the Train Data Set**
# %%
del som_4X4
del som_20X20

som_4X4   = MiniSom(4,  4,  64, sigma=3, learning_rate=0.5, neighborhood_function='triangle')
som_20X20 = MiniSom(20, 20, 64, sigma=4, learning_rate=0.5, neighborhood_function='triangle')

# %%
#trainning SOM Models on All Data
som_4X4.train  (train_data, 10000, verbose=True)
som_20X20.train(train_data, 10000, verbose=True)

# %% [markdown]
# **Assigning label to each neurons using majority voting**
# %%
#
#   4X4
#
voting_list_lable_4X4 = np.zeros((4,4,10), dtype=int);
for x, t in zip(train_data, train_label):
    w = som_4X4.winner(x)
    voting_list_lable_4X4[w[0]][w[1]][t] += 1
Neuron_lable_4X4 = np.zeros((4,4), dtype=int);
for i in range(4):
    for j in range(4):
        Neuron_lable_4X4[i][j] = np.argmax(voting_list_lable_4X4[i][j])

#
#   20X20
#
voting_list_lable_20X20 = np.zeros((20,20,10), dtype=int);
for x, t in zip(train_data, train_label):
    w = som_20X20.winner(x)
    voting_list_lable_20X20[w[0]][w[1]][t] += 1
Neuron_lable_20X20 = np.zeros((20,20), dtype=int);
for i in range(20):
    for j in range(20):
        Neuron_lable_20X20[i][j] = np.argmax(voting_list_lable_20X20[i][j])

# %% [markdown]
# **Classifying the Test Data**
# %%
#
#   4X4
#
#   Train Dataset
train_correctly_classified_cntr_4X4 = 0;
im = 0;
for x in train_data:
    w = som_4X4.winner(x)
    train_correctly_classified_cntr_4X4 += (Neuron_lable_4X4[w[0]][w[1]] == train_label[im])
    im += 1

#   Test Dataset
test_label_classified_4X4 = np.zeros((len(test_data),), dtype=int);
test_correctly_classified_cntr_4X4 = 0;
im = 0;
for x in test_data:
    w = som_4X4.winner(x)
    test_label_classified_4X4[im] = Neuron_lable_4X4[w[0]][w[1]]
    test_correctly_classified_cntr_4X4 += (test_label_classified_4X4[im] == test_label[im])
    im += 1



#
#   20X20
#
#   Train Dataset
train_correctly_classified_cntr_20X20 = 0;
im = 0;
for x in train_data:
    w = som_20X20.winner(x)
    train_correctly_classified_cntr_20X20 += (Neuron_lable_20X20[w[0]][w[1]] == train_label[im])
    im += 1

#   Test dataset
test_label_classified_20X20 = np.zeros((len(test_data),), dtype=int);
test_correctly_classified_cntr_20X20 = 0;
im = 0;
for x in test_data:
    w = som_20X20.winner(x)
    test_label_classified_20X20[im] = Neuron_lable_20X20[w[0]][w[1]]
    test_correctly_classified_cntr_20X20 += (test_label_classified_20X20[im] == test_label[im])
    im += 1

# %% [markdown]
# **Evaluating the Classification Performance**
# %%
#
#   4X4
#
print("Accuracy of 4X4 neuron SOM model on train data is: ", train_correctly_classified_cntr_4X4/len(train_data)*100, "%")
print("Accuracy of 4X4 neuron SOM model on test  data is: ",  test_correctly_classified_cntr_4X4/len( test_data)*100, "%\r\n")

conf_matrix_4X4 = np.zeros((10,10), dtype=int);
for i in range(len(test_label)):
    conf_matrix_4X4[test_label[i]][test_label_classified_4X4[i]] += 1;

print("Confusion Maxrix of 4X4 neuron SOM model is: ")
print(conf_matrix_4X4)

TP_4X4 = np.zeros((10,), dtype=int);
FP_4X4 = np.zeros((10,), dtype=int);
FN_4X4 = np.zeros((10,), dtype=int);
TN_4X4 = np.zeros((10,), dtype=int);
for i in range(10):
    TP_4X4[i] = conf_matrix_4X4[i][i]
    FP_4X4[i] = sum(np.transpose(conf_matrix_4X4)[i]) - TP_4X4[i];
    FN_4X4[i] = sum(conf_matrix_4X4[i]) - TP_4X4[i];
    TN_4X4[i] = sum(sum(conf_matrix_4X4)) - TP_4X4[i] - FP_4X4[i] - FN_4X4[i];
print("")
print("TP of 4X4 Neuron SOM Model: ", TP_4X4)
print("FP of 4X4 Neuron SOM Model: ", FP_4X4)
print("FN of 4X4 Neuron SOM Model: ", FN_4X4)
print("TN of 4X4 Neuron SOM Model: ", TN_4X4)

Precision_4X4 = TP_4X4/(TP_4X4+FP_4X4)
Recall_4X4 = TP_4X4/(TP_4X4+FN_4X4)
F1_4X4 = 2*Precision_4X4*Recall_4X4/(Precision_4X4+Recall_4X4)

print("")
print("Precision of 4X4 Neuron SOM Model: ",  [ '%.3f' % elem for elem in Precision_4X4 ])
print("Recall    of 4X4 Neuron SOM Model: ", [ '%.3f' % elem for elem in Recall_4X4 ])
print("F1-score  of 4X4 Neuron SOM Model: ", [ '%.3f' % elem for elem in F1_4X4 ])

print("")
print("Mean of Precision of 4X4 Neuron SOM Model Over all Classes: ", f"{np.nan_to_num(Precision_4X4).mean():.4f}" )
print("Mean of Recall    of 4X4 Neuron SOM Model Over all Classes: ", f"{np.nan_to_num(Recall_4X4).mean():.4f}"    )
print("Mean of F1-score  of 4X4 Neuron SOM Model Over all Classes: ", f"{np.nan_to_num(F1_4X4).mean():.4f}"        )
print("\r\n\r\n\r\n")

#
#   20X20
#
print("Accuracy of 20X20 neuron SOM model on train data is: ", train_correctly_classified_cntr_20X20/len(train_data)*100, "%")
print("Accuracy of 20X20 neuron SOM model on test  data is: ",  test_correctly_classified_cntr_20X20/len( test_data)*100, "%\r\n")

conf_matrix_20X20 = np.zeros((10,10), dtype=int);
for i in range(len(test_label)):
    conf_matrix_20X20[test_label[i]][test_label_classified_20X20[i]] += 1;

print("Confusion Maxrix of 20X20 neuron SOM model is: ")
print(conf_matrix_20X20)

TP_20X20 = np.zeros((10,), dtype=int);
FP_20X20 = np.zeros((10,), dtype=int);
FN_20X20 = np.zeros((10,), dtype=int);
TN_20X20 = np.zeros((10,), dtype=int);
for i in range(10):
    TP_20X20[i] = conf_matrix_20X20[i][i]
    FP_20X20[i] = sum(np.transpose(conf_matrix_20X20)[i]) - TP_20X20[i];
    FN_20X20[i] = sum(conf_matrix_20X20[i]) - TP_20X20[i];
    TN_20X20[i] = sum(sum(conf_matrix_20X20)) - TP_20X20[i] - FP_20X20[i] - FN_20X20[i];
print("")
print("TP of 20X20 Neuron SOM Model: ", TP_20X20)
print("FP of 20X20 Neuron SOM Model: ", FP_20X20)
print("FN of 20X20 Neuron SOM Model: ", FN_20X20)
print("TN of 20X20 Neuron SOM Model: ", TN_20X20)

Precision_20X20 = TP_20X20/(TP_20X20+FP_20X20)
Recall_20X20 = TP_20X20/(TP_20X20+FN_20X20)
F1_20X20 = 2*Precision_20X20*Recall_20X20/(Precision_20X20+Recall_20X20)

print("")
print("Precision of 20X20 Neuron SOM Model: ", [ '%.3f' % elem for elem in Precision_20X20 ])
print("Recall    of 20X20 Neuron SOM Model: ", [ '%.3f' % elem for elem in Recall_20X20 ])
print("F1-score  of 20X20 Neuron SOM Model: ", [ '%.3f' % elem for elem in F1_20X20 ])

print("")
print("Mean of Precision of 20X20 Neuron SOM Model Over all Classes: ", f"{Precision_20X20.mean():.4f}" )
print("Mean of Recall    of 20X20 Neuron SOM Model Over all Classes: ", f"{Recall_20X20.mean():.4f}"    )
print("Mean of F1-score  of 20X20 Neuron SOM Model Over all Classes: ", f"{F1_20X20.mean():.4f}"        )

# %% [markdown]
# **Visualization**
# %%
#
#   4X4
#
Vwmap_4X4 = {}
for x in train_data:
    w = som_4X4.winner(x)
    Vwmap_4X4[w] = 1

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
all_patches = []
cmap = cm.get_cmap('viridis')
for i in range(4):
    for j in range(4):
        rect_color = cmap(int((256*Neuron_lable_4X4[j][i])/9))
        text_color = 'white' if rect_color[0] < 0.4 else 'black' # Flip the color for better visibility
        if (i, j) in Vwmap_4X4:
            ax.add_patch( patches.Circle((i+0.5, j+0.5), 0.4, linewidth=1, fill=1, edgecolor='b' , facecolor=rect_color,))
            plt. text(i+0.37,  j+0.3725,  str(Neuron_lable_4X4[j][i]), color=text_color, fontdict={'weight': 'bold',  'size': 22})
        else:
            ax.add_patch( patches.Circle((i+0.5, j+0.5), 0.4, linewidth=1, fill=1, edgecolor='b' , facecolor='white'))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=9))
plt.colorbar(sm, ax=ax)
plt.axis([-0.5, som_4X4.get_weights().shape[0]+0.5, -0.5,  som_4X4.get_weights().shape[1]+0.5])
plt.savefig('outputs/figures/SOM_4X4_Neurons_label.png')
plt.show()
#print(Neuron_lable_4X4)



#
#   20X20
#
Vwmap_20X20 = {}
for x in train_data:
    w = som_20X20.winner(x)
    Vwmap_20X20[w] = 1

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
all_patches = []
cmap = cm.get_cmap('viridis')
for i in range(20):
    for j in range(20):
        rect_color = cmap(int((256*Neuron_lable_20X20[j][i])/9))
        text_color = 'white' if rect_color[0] < 0.4 else 'black' # Flip the color for better visibility
        if (i, j) in Vwmap_20X20:
            ax.add_patch( patches.Circle((i+0.5, j+0.5), 0.4, linewidth=1, fill=1, edgecolor='b' , facecolor=rect_color,))
            plt. text(i+0.225,  j+0.325,  str(Neuron_lable_20X20[j][i]), color=text_color, fontdict={'weight': 'bold',  'size': 8})
        else:
            ax.add_patch( patches.Circle((i+0.5, j+0.5), 0.4, linewidth=1, fill=1, edgecolor='b' , facecolor='white'))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=9))
plt.colorbar(sm, ax=ax)
plt.axis([-0.5, som_20X20.get_weights().shape[0]+0.5, -0.5,  som_20X20.get_weights().shape[1]+0.5])
plt.savefig('outputs/figures/SOM_20X20_Neurons_label.png')
plt.show()
#print(Neuron_lable_20X20)

