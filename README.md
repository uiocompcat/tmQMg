**Update 2024**: The tmQMg dataset has been extended by 13,756 transition metal complexes extracted from the Cambridge Structural Database.

# tmQMg

This repository contains the graph dataset tmQMg containing descriptive graph representations of 74,636 transition metal complexes (TMCs), including all thirty elements from the 3d, 4d, and 5d series. These representations were derived from quantum chemistry simulation data and more preciseley Natural Bond Order (NBO) analysis. We provide three different types of graphs as GML formatted files: baseline, u-NatQG and d-NatQG. The graphs can be used in deep graph learning methods and can be downloaded from [here](https://archive.sigma2.no/pages/public/datasetDetail.jsf?id=10.11582/2025.00007). The code used to generate these representations can be found at [HyDGL](https://github.com/hkneiding/HyDGL). A detailed discussion about the representations and machine learning methods can be found in the corresponding [publication](https://doi.org/10.1039/D2DD00129B).

![tmQMg_Figure](tmQMg.png)

## Data

###### [data/tmQMg_graphs.md](data/tmQMg_graphs.md)
- Overview of the different graph types and links to their storage location.

###### [data/tmQMg_properties_and_targets.csv](data/tmQMg_properties_and_targets.csv)
- List of all TMCs and their respective graph level features and quantum properties.
- Graph level features are: charge, molecular mass, number of atoms and number of electrons

###### [data/tmQMg_xyz.zip ](data/tmQMg_xyz.zip)
- Zip file of the xyz data of all compounds in the dataset.

## Code

Furthermore, we provide here the Python codes used to perform the various machine learning experiments.

###### [scripts/outliers.txt](scripts/outliers.txt)
- List of the IDs of about 2.5k of the TMCs that were deemed to be outliers based on their quantum properties for the performed ML experiments.

###### [scripts/Gilmer-MPNN/](scripts/Gilmer-MPNN/)
- Holds the code for the Gilmer net and comprehensive analysis of data.
- Consult the provided README for more info.
