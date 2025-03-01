# EUSFLAT2025
Code for the **(alpha,beta)-SRBMs classifier** appearing in the paper:

D. Petturiti and M. Rifqi (2025).
_Alpha-Maxmin Classification with an Ensemble of Structural Restricted Boltzmann Machines_.

# FILE INVENTORY

**SRBM_classifier.py:** Class that realizes a Structural Restricted Boltzmann Machine (SRBM) classifier, with a given percentage of structural zeros (missing hedges in the bipartite graph).

**alpha_beta_SRBMs_classifier.py:** Credal classifier relying on a heterogeneous ensemble of SRBMs, where classification is carried out with alpha-maxmin decision criterion and beta-quantile filtering of outliers.
