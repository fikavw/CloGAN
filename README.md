# CloGAN
Implementation of Rios, A., & Itti, L. (2019, August). Closed-loop memory gan for continual learning. In Proceedings of the 28th International Joint Conference on Artificial Intelligence (pp. 3332-3338). AAAI Press.

Uses pytorch:
Torch version 0.4.1

To train CloGAN type:

python3 -i CloGAN_main.py --dataset_name DATASETNAME --dataroot PATH_TO_DATASET --main_out_path PATH_TO_RESULTS --fraction_buff BUFFER_USAGE --save_generated_images

Where the following flags should be modified appropriately:

1. BUFFER_USAGE: Set buffer usage as percentage of dataset (ex: 0.01 = 1% of entire dataset) to store:
--fraction_buff 0.01 (ex for svhn 1% or 5%)
--fraction_buff 0.01 (ex for emnist 1% or 5%)
--fraction_buff 0.01 (ex for mnist 0.1% or 0.5%)
--fraction_buff 0.01 (ex for fashion 0.1% or 0.5%)

2. DATASETNAME: Define which dataset to use:
--dataset_name mnist
--dataset_name fashion
--dataset_name emnist
--dataset_name svhn

3. PATH_TO_DATASET: Set path to where dataset is stored
--dataroot /home/data/ (for example)

4. PATH_TO_RESULTS: Set path to general results folder
--main_out_path /home/CloGAN/results/
