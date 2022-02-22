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

COLUMNS = ['mean_ax', 'std_ax', 'mean_my', 'std_my', 'mean_bg', 'std_bg']

def get_stats_from_histogram(histogram):
    '''
    This function computes mean and standard deviation values from an histogram.
    '''
    size = np.array(list(histogram.values())).sum() 
    mean = 0
    for (val, count) in histogram.items():
        mean += val * count / size
    variance = 0
    for (val, count) in histogram.items():
        variance += count * (val - mean)**2 / size
    std = np.sqrt(variance)
    return [mean, std]

def analyze_image(img, axon_mask, myelin_mask, saving=False):
    '''
    This function computes histograms for the axon, myelin and backgound classes
    and returns mean and standard deviation values for all classes.
    '''
    bg_mask = 255 - (axon_mask + myelin_mask)
    nb_axon_px = (axon_mask == 255).sum()
    nb_myelin_px = (myelin_mask == 255).sum()
    nb_bg_px = (bg_mask == 255).sum()
    total_size = nb_axon_px + nb_myelin_px + nb_bg_px

    # filter input image with masks
    axon_only = img * axon_mask
    myelin_only = img * myelin_mask
    bg_only = img * bg_mask

    # compute histograms and remove non-class pixels (important for means and stds)
    u_axon, c_axon = np.unique(axon_only, return_counts=True)
    u_myelin, c_myelin = np.unique(myelin_only, return_counts=True)
    u_bg, c_bg = np.unique(bg_only, return_counts=True)
    hist_axon = dict(zip(u_axon, c_axon))
    hist_myelin = dict(zip(u_myelin, c_myelin))
    hist_bg = dict(zip(u_bg, c_bg))
    hist_axon[0] -= total_size - nb_axon_px
    hist_myelin[0] -= total_size - nb_myelin_px
    hist_bg[0] -= total_size - nb_bg_px

    histograms = [hist_axon, hist_myelin, hist_bg]
    measures = []
    for hist in histograms:
        stats = get_stats_from_histogram(hist)
        measures.append(stats[0])
        measures.append(stats[1])
    print(measures)
    measures = pd.DataFrame([measures], columns=COLUMNS)
    print(measures)


    if saving:
        plt.figure(1)
        plt.plot(hist_axon.keys(), hist_axon.values())
        plt.title('Histogram for axon')
        plt.savefig("hist_axon")    
        print("Axon histogram saved.")
        plt.figure(2)
        plt.plot(hist_myelin.keys(), hist_myelin.values())
        plt.title('Histogram for myelin')
        plt.savefig("hist_myelin")    
        print("Myelin histogram saved.")
        plt.figure(3)
        plt.plot(hist_bg.keys(), hist_bg.values())
        plt.title('Histogram for background')
        plt.savefig("hist_bg")    
        print("Background histogram saved.")

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
    print("mean value= ", np.mean(img))
    print("std value= ", np.std(img))

    maskpath = dataset_path / "derivatives" / "labels" / subj_list[0] / "micr"
    masks = maskpath.glob('*.png')
    for m in masks:
        if "axon-manual" in str(m):
            axon = ads_utils.imread(m)
        elif "myelin-manual" in str(m):
            myelin = ads_utils.imread(m)

    analyze_image(img, axon, myelin)