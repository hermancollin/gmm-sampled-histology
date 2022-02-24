"""
Script generating a bunch of synthetic data from the segmentation masks.
"""
from AxonDeepSeg import ads_utils
from pathlib import Path
import sys
import json
import os
import time
import numpy as np
from gen_model import ImageGenerator


def build_label_map(axon_mask, myelin_mask):
    """
    This function creates a label map with values 0,1 and 2 for background, 
    myelin and axon, respectively.
    """
    label_map = np.zeros(axon_mask.shape, dtype=np.int8)
    myelin_x, myelin_y = np.where(myelin_mask > 128)
    for x,y in zip(myelin_x, myelin_y):
        label_map[x, y] = 1
    axon_x, axon_y = np.where(axon_mask > 128)
    for x,y in zip(axon_x, axon_y):
        label_map[x, y] = 2
    return label_map
    

def main():
    with open('dataset_atlas.json', 'r') as json_file:
        atlas = json.load(json_file)
    gmm = ImageGenerator()
    for sample in atlas.keys():
        axon_mask = ads_utils.imread(atlas[sample]["axon_mask"])
        myelin_mask = ads_utils.imread(atlas[sample]["myelin_mask"])
        label_map = build_label_map(axon_mask, myelin_mask)

        start_time = time.time()
        img = gmm.generate_image(label_map)
        end_time = time.time()
        print(f"Image generated in {end_time - start_time} seconds.")
        output_fname = Path('.') / "output" / (sample + '_synthetic.png')
        ads_utils.imwrite(output_fname, img)


if __name__ == '__main__':
    main()