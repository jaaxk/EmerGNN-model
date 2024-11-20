# Our adaptation of EmerGNN [(Zhang et. al)](https://www.nature.com/articles/s43588-023-00558-4) to simplify inference making
## Description
- We trained the original model using the S0 dataset (interactions between existing drugs) on GPU and provide the states here as `S0_saved_model.pt`.  
- We created `make_inferences.py` which makes inferences given two drugs (CID, DrugBank ID, or Smiles) and returns the relationships (concept ID) in a list.
  - This script uses the `id2drug.json` file provided in the original EmerGNN repository and the `id2relation.json` file provided in the [KnowDDI GitHub repository](https://github.com/LARS-research/KnowDDI).  
- We modified the scripts from the TWOSIDES directory of the [original GitHub repository](https://github.com/LARS-research/EmerGNN) to work with our script on CPU.
- This purpose of this repository is to be be incorporated into [our web app](https://github.com/Orbin-Ahmed/EmergnnWebUI) for a simple UI to predict DDIs.

## Running the code  
First clone the repository, and create a conda environment with python 3.9  
```
conda create -n emergnn python==3.9
```
Then install dependencies
```
pip install -r requirements.txt
```
Then modify the script `make_inferences.py` to input the drugs of interest.  

The easiest way to find DDIs, however, is to use our website: https://github.com/Orbin-Ahmed/EmergnnWebUI
## Credit
This model came from the paper by [Zhang et. al, DOI: 10.5281/zenodo.10017431](https://www.nature.com/articles/s43588-023-00558-4)  
And the respective repository [LARS-Research/EmerGNN](https://github.com/LARS-research/EmerGNN)
