# OCELOTML 2D
This repository contains the code to download the OCELOT chromophore dataset and use the pretrained models to make predictions. You can find the online implementation at [here](https://oscar.as.uky.edu/ocelotml_2d)



## Files
`eval.ipynb` - make predictions with SMILES

`MPNN_evidential` - the MPNN model with evidential deep learning

`dataset.ipynb` - download the OCELOT chrmomophore v1 dataset and transform to pandas DataFrame

`mlp_features` and `normalize_feats.csv` - generate the features for model input

`first_gen_models.ipynb` - Training of first generation model

## Citing
If you use the dataset or any trained models in your work, please cite the following article-

Bhat, V.; Sornberger, P.; Pokuri, B. S. S.; Duke, R.; Ganapathysubramanian, B.; Risko, C. Electronic, Redox, and Optical Property Prediction of Organic P-Conjugated Molecules through a Hierarchy of Machine Learning Approaches. _Chemical Science_ 2022. [https://doi.org/10.1039/d2sc04676h](https://doi.org/10.1039/d2sc04676h).
