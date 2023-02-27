## Gilmer MPNN

This directory holds the code for the experiments conducted with the Gilmer architecture. It is designed to be executed using the Python package Weights and biases (wandb) to log the results.

### Requirements

You need a Python3.9 installation with the following packages:

- [wandb](https://wandb.ai/site)
- [pytorch](https://pytorch.org/)
- [pytorch_geometric](https://www.pyg.org/)
- [HyDGL](https://github.com/hkneiding/HyDGL)

Because logging is done using the [wandb](https://wandb.ai/site) package you will need an account to run these scripts.

### Use
Navigate into this directory and open the file ``ml.py``. In this file, edit the entries <wandb_project_name> and <wandb_entity> with your wandb credentials. This will direct all the logging output to a project in you wandb account. Then edit the entry <root_dir> with the path of the directory on you machine that you want to store the raw and processed data in.

After that, run with ``python3 ml.py`` which will run models for the baseline, u-NatQG and d-NatQG for all properties investigated in the paper. These are:

- HOMO-LUMO gap
- Polarizability
- Dipole moment
- HOMO energy
- LUMO energy
- Electronic energy
- Dispersion energy
- Zero-point energy
- Enthalpy energy
- Heat capacity
- Entropy energy
- Gibbs energy
- Thermodynamic correction
- Largest vibrational frequency

If you want to only run the models for a subset of properties and representations you can change the corresponding function calls at the very bottom of the ``ml.py`` script. The models will use the hyperparameters reported in the paper but you can change the settings in the corresponding functions, i.e. ``run_baseline()``, ``run_uNatQ()`` and ``run_dNatQ()``.

Note that the script ``tmQMg.py`` will automatically download all necessary files from the tmQMg repository. You can also use it to build your own machine learning pipeline using different models.