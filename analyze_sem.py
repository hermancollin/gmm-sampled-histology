"""
This script analyzes the SEM dataset and collects information on pixel intensity 
class-wise. It outputs aggregated mean and standard deviation values for every class
(background, axon and myelin) 

"""

from AxonDeepSeg import ads_utils
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# locating data and listing participants
dataset_path = Path("/home/herman/Documents/NEUROPOLY_21/datasets/data_axondeepseg_sem")
participants_path = dataset_path / 'participants.tsv'
with open(participants_path) as tsv_file:
    subj_list = pd.read_csv(participants_path, delimiter='\t')
subj_list = subj_list["participant_id"]
print(subj_list)

# loading data
#for subject in subj_list:
#    data_path = dataset_path / subject / "micr"
#    mask_path = dataset_path / "derivatives" / "labels" / subject / "micr"
#    files = data_path.glob('*.png')
#    for f in files:
#        print(f)
#    masks = mask_path.glob('*axonmyelin*')
#    for m in masks:
#        print(m)

first_subj = dataset_path / subj_list[0] / "micr"
files = first_subj.glob('*png')
for file in files:
    print(file)
    img = ads_utils.imread(file)
    print("Image dimensions: ", img.shape)
    unique, counts = np.unique(img, return_counts=True)
    stats = dict(zip(unique, counts))
    print("Histogram saved for ", file)
    plt.plot(stats.keys(), stats.values())
    plt.savefig("sub-rat1_histogram")
    print("mean value= ", np.mean(img))
    print("std value= ", np.std(img))