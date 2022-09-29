## Gilmer MPNN

This directory holds the code for the experiments conducted with the Gilmer architecture. It is designed to be executed using the Python package Weights and biases (wandb) to log the results. 

### Requirements

You need a Python3 installation with the following packages:

- [wandb](https://wandb.ai/site)
- [pytorch](https://pytorch.org/)
- [pytorch_geometric](https://www.pyg.org/)
- [HyDGL](https://github.com/hkneiding/HyDGL)

### Use
Navigate into this directory and open the file 'ml.py'. In this file edit the entries <wandb_project_name> and <wandb_entity> with your wandb credentials. Then edit the entry <root_dir> with the directory on you machine that you want to store the raw and processed data in.

After that, run with ``python3 ml.py``. 

Note that the script ``tmQMg.py`` will download all necessary files automatically from the tmQMg repository. You can also use it to build your own machine learning pipeline.