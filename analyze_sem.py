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
import json

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
    # binarization
    axon_mask = axon_mask > 128
    myelin_mask = myelin_mask > 128
    bg_mask = bg_mask > 128

    nb_axon_px = (axon_mask == True).sum()
    nb_myelin_px = (myelin_mask == True).sum()
    nb_bg_px = (bg_mask == True).sum()
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
    measures = pd.DataFrame([measures], columns=COLUMNS)
    #print(measures)

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
        #ads_utils.imwrite('axon_only.png', axon_only)
        #ads_utils.imwrite('myelin_only.png', myelin_only)
        #ads_utils.imwrite('bg_only.png', bg_only)
    return measures


# locating data and listing samples
dataset_path = Path("/home/herman/Documents/NEUROPOLY_21/datasets/data_axondeepseg_sem")
samples_path = dataset_path / 'samples.tsv'
samples = pd.read_csv(samples_path, delimiter='\t')
samples_dict = {}
for i, row in samples.iterrows():
    subject = row['participant_id']
    sample = row['sample_id']
    if sample not in samples_dict:
        samples_dict[sample] = {}
    samples_dict[sample]['subject'] = subject

# loading data
for sample in samples_dict.keys():
    subject = samples_dict[sample]['subject']
    data_path = dataset_path / subject / "micr"
    mask_path = dataset_path / "derivatives" / "labels" / subject / "micr"
    files = data_path.glob('*.png')
    for f in files:
        if sample in str(f):
            samples_dict[sample]['image'] = str(f)
    mask_files = list(mask_path.glob('*-axon-*')) + list(mask_path.glob('*-myelin-*'))
    for m in mask_files:
        if sample in str(m):
            if 'seg-axon' in str(m):
                samples_dict[sample]['axon_mask'] = str(m)
            elif 'seg-myelin' in str(m):
                samples_dict[sample]['myelin_mask'] = str(m)
print(f'Found {len(samples_dict.keys())} samples:')
print(json.dumps(samples_dict, indent=4))
with open('dataset_atlas.json', 'w') as json_file:
    json.dump(samples_dict, json_file)

# extracting mean and std from all samples
priors = pd.DataFrame(columns=COLUMNS)
for sample in samples_dict.keys():
    image = ads_utils.imread(samples_dict[sample]['image'])
    axon_mask = ads_utils.imread(samples_dict[sample]['axon_mask'])
    myelin_mask = ads_utils.imread(samples_dict[sample]['myelin_mask'])
    
    row = analyze_image(image, axon_mask, myelin_mask)
    row.index = [sample]
    priors = priors.append(row)
priors.to_csv('priors.csv')