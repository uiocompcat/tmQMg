# Radius Graph Models for tmqmg dataset (Dimenet++, SchNet, EdgeUpdateNet)
## Installation

To run SOTA radius graph based models you need to install qsar-flash package by cloning the repository [qsar-flash] and installing dependencies into your conda environment:

```sh
git clone https://github.com/AI4Materials-lab/qsar_flash.git
cd qsar_flash
pip install conda-lock
pip install poetry
poetry install
```
## Train and validate
To run train and validation of models (Dimenet++, SchNet, EdgeUpdateNet) choose 'qsar_flash/scripts/tmqm directory'. Set the path to your local 'data/tmQMg_xyz' folder withing the following .py files and run one of the following commands:

```sh
python dimenet_plus_plus_train.py
python edge_update_net.py
python schnet_train.py
```


   [qsar-flash]: <https://github.com/AI4Materials-lab/qsar_flash.git>
   