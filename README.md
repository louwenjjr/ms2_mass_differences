# ms2_mass_differences
Identify meaningful mass differences in MS2 spectra to include in spectral embeddings/substructure models.

To recreate this analysis it is best to create a conda environment like such:
```
conda create --name spec_analysis python=3.8
conda activate spec_analysis
conda install --channel nlesc --channel bioconda --channel conda-forge spec2vec
conda install -c conda-forge rdkit
conda install ipython
pip install jupyter
```
