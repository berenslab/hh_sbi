# Simulator-based inference for Hodgkin-Huxley-based models.
This repository presents the code for the manuscript [insert].

## Raw data
Raw data can be publicly downloaded from [here](https://dandiarchive.org/dandiset/000008/draft) corresponding to [this](https://www.nature.com/articles/s41586-020-2907-3) study published in Nature. Instructions on how to do so with `dandi` are also found there. Make sure you download the data in `./data/raw_data`.

## Requirements
[dandi](https://dandiarchive.org/) (see `raw data`) <br>
[pynwb](https://pynwb.readthedocs.io/en/stable/): `pip install -U pynwb` tested with version 2.2.0 <br> 
[brian2](https://brian2.readthedocs.io/en/stable/): `pip install brian2` tested with version 2.5.0.3 <br>
[sbi](https://www.mackelab.org/sbi/reference/): `pip install sbi` tested with version 0.21.0 <br>
[pathos](https://github.com/uqfoundation/pathos): `pip install pathos` tested with version 0.2.8 <br>
[openTSNE](https://opentsne.readthedocs.io/en/latest/installation.html#conda): `pip install opentsne` tested with version 0.6.2 <br>