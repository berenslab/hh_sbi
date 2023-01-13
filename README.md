# Simulator-based inference for Hodgkin-Huxley-based models.
This repository presents the code for the manuscript [insert].

## Raw data
Raw data can be publicly downloaded from [here](https://dandiarchive.org/dandiset/000008/draft) corresponding to this [study](https://www.nature.com/articles/s41586-020-2907-3) in Nature. Make sure you are in the `./data/raw_data` directory when you use `dandi` in your command line.

## Requirements
[dandi](https://dandiarchive.org/) (see `raw data`) <br>
[pynwb](https://pynwb.readthedocs.io/en/stable/): `pip install -U pynwb` tested with version 2.2.0 <br> 
[brian2](https://brian2.readthedocs.io/en/stable/): `pip install brian2` tested with version 2.5.0.3 <br>
[sbi](https://www.mackelab.org/sbi/reference/): `pip install sbi` tested with version 0.21.0 <br>
[pathos](https://github.com/uqfoundation/pathos): `pip install pathos` tested with version 0.2.8 <br>
