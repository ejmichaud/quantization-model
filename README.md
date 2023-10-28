# quantization-model
Cleaned up code for reproducing the NeurIPS 2023 version of our paper "The Quantization Model of Neural Scaling": https://arxiv.org/abs/2303.13506

## Organization
```
.
├── experiments
├── figures
├── LICENSE
├── notebooks
├── phase-changes
├── README.md
├── scaling-laws.csv
├── scripts
├── texts
└── tinystories
```

Scripts defining experiments (e.g. slurm job arrays for grid searches) are in `experiments`. Paper figures are saved to `figures`. Notebooks for creating these figures are in `notebooks`. In `scripts` we define the code for training networks on multitask sparse parity. We save text snippets (samples from QDG clusters) to `texts`. Code for applying QDG to the TinyStories dataset and saving results is in `tinystories`. `scaling-laws.csv` is the database provided by Epoch from their scaling laws literature review: https://epochai.org/blog/scaling-laws-literature-review

## Reproducing each figure

Here are rough instructions for reproducing most of the paper's figures. Note that these are not ready to be run: you will need to modify each to e.g. load up data from the correct location on your system, save to the correct location on your system, etc. I have copied these over from a much messier repo, so the paths still reference the name of that old repo, which was `the-everything-machine`. It is possible that I have left out major steps in reproducing these from the desciptions below -- feel free to email me at ericjm [at] mit.edu if you have any questions. 

**Figure 1** and **Figure 13**: Text snippets created in `notebooks/save-clusters.ipynb`. Running this notebook requires several experiments to be run first. First, one needs to download the test set of The Pile, `test.jsonl.zst`, either from https://pile.eleuther.ai/ or from [here](https://www.dropbox.com/scl/fi/njeocnzo8wzfeep8clm0z/test.jsonl.zst?rlkey=gz68ewdcyktfcekd7pz3n1xcx&dl=0). Then we must create our canonical tokenization of the dataset (which will allow us to consistently map integers to tokens in `test.jsonl.zst`, which can be done with `scripts/create_pile_canonical.py`. In addition to this, the notebook requires the `clusters_full_more.pkl` file containing the clusters from spectral clustering as well as the `full_more.pt` file containing the Pile test set token indices that were used by QDG. These can be downloaded [here](https://www.dropbox.com/scl/fi/87eq1e6q59kuprimlzbtu/clusters_full_more.pkl?rlkey=5lfwf8grnhkp4af6v0vsbpkv4&dl=0) and [here](https://www.dropbox.com/scl/fi/mlm6jzjghcbcw7lxmqlww/full_more.pt?rlkey=s8y3sgipwimabxa87qj6g4dqh&dl=0), respectively. If you want to run QDG to create these yourself, there are several steps. The `full_more.pt` file is created by `experiments/clustering-0/compute_similarity_full_more.py`. This script requires the `zero_and_induction_idxs.pkl` file. This file contains indices of tokens in the test set of the Pile where `pythia-19m` achieves less than 0.1 nats of cross-entropy and indices of tokens which are potentially predictable just via induction from their context (they are the third token in a trigram that occurred earlier in the context) -- we attempt to filter out these tokens which can be achieved via induction since for a small model like `pythia-19m`, it seems like a significant fraction of tokens on which the model achieves very low loss can be predicted in this way, would would make it harder to discover other quanta. The `zero_and_induction_idxs.pkl` file can be downloaded [here](https://www.dropbox.com/scl/fi/v2et8npxbhnsym0d3c5n6/zero_and_induction_idxs.pkl?rlkey=fedbwii5dp560vtq81cws3yh8&dl=0) or created yourself with the `scripts/zero_and_induction_idxs.py` script. Note that this script requires the `pythia-2.npy` file, for which the instructions to download or create are below (for Figure 3). The `clusters_full_more.pkl` is created by `experiments/clustering-0/compute_clusters_full_more.py`.

**Figure 2** - `figures/parameters-steps-data-emergence-and-scaling-scalingtop.png`: Created in `notebooks/combined-scaling-and-emergence-plots.ipynb`, using data from `experiments/P-scaling-15` and `experiments/D-scaling-6`

**Figure 3** - `figures/pythia-scaling-tripanel.png` and **Figure 11** - `figures/pythia-dynamics-tripanel.png`: Created in `notebooks/ten-million-scaling-curves.ipynb`. These figures use data from the following files: `pythia-2.npy`, `timeseries19m.npy`, `timeseries125m.npy`, `timeseries350m.npy`, `timeseries800m.npy`, `timeseries1_3b.npy`. The scripts for creating `pythia-2.npy` are in `experiments/pythia-2` and the scripts for creating the timeseries files are in `./phase-changes`. The notebook also loads up a `num_params.pt` file, which is created with the `experiments/pythia-2/get_num_params.py` script. Note that all these experimentes used the `-v0` version of the Pythia models before the naming convention was changed. So e.g. `pythia-19m` refers to what today would be called `pythia-70m-v0`. If you just want to reproduce Figure 3 without recomputing the losses yourself, you can download the `pythia-2.npy` file from my Dropbox [here](https://www.dropbox.com/scl/fi/oopaiad41vkz6iep1dscu/pythia-2.npy?rlkey=f6f4nbvvdr83hwsu9u50xsdfr&dl=0) and get the `num_params.pt` file [here](https://www.dropbox.com/scl/fi/o03hqi7oqktys4wn36tq8/num_params.pt?rlkey=m3kyudrqs90dqcacb3atf6324&dl=0)

**Figure 4** - `figures/tokenscaling/tokensinghsirsa.pdf`, `figures/tokenscaling/tokenfruit-influx.pdf` and **Figure 12** - `figures/tokenscaling/tokenneilmackinnon.pdf`, `figures/tokenscaling/tokenssep-normal.pdf`, `figures/tokenscaling/tokenessmarshall.pdf`, `figures/tokenscaling/tokenonconsumer.pdf`: Created in `notebooks/diverse-scaling-examples.ipynb`. Note that this notebook requires both a tokenized version of The Pile (discussed above for Figure 1) and the `pythia-2.npy` curves data, which, again, can be downloaded from Dropbox [here](https://www.dropbox.com/scl/fi/oopaiad41vkz6iep1dscu/pythia-2.npy?rlkey=f6f4nbvvdr83hwsu9u50xsdfr&dl=0).

**Figure 15** - `figures/similarity-matrix-and-rank-frequency-envelope.png`: Created in `notebooks/llm-clustering-plot.ipynb`. Requires that the similarity matrix and clusters have already been computed or downloaded. See the instructions from making Figure 1. 

**Figure 7** and **Figure 8** - `figures/sparse-parity-convergence-time.pdf`: Created in `notebooks/sparse-parity-timeseries.ipynb`. 

**Figure 9** - `figures/sparse-parity-data-scaling-dependence-n.pdf` and **Figure 10** - `figures/sparse-parity-all-scaling-varying-alpha.pdf`. Created by `notebooks/scaling-exp-vs-zipf-exp.ipynb`.

**Figure 14**: Text snippets created in `tinystories/save-clusters.ipynb`. But to run this, you first need to run `tinystories/scratch.ipynb` to compute  a `losses.pt` file (losses of TinyStories-33M across tokens in TinyStories). Then run `tinystories/sim_matrix.ipynb` to compute the similarity matrix for QDG, then run `tinystories/spectral_clustering.ipynb` to compute the clusters.

**Figure 18** - `figures/scaling-scatter-linear-scale.pdf`: Created with `notebooks/scaling-exponents-scatter.ipynb` using `./scaling-laws.csv`


