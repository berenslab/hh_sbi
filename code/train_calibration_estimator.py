from fslm4expdata import BiasEstimator
import wandb
import argparse
from fslm.experiment_helper import SimpleDB
from fslm.utils import includes_nan 

from numpy.random import seed as npseed
from torch import manual_seed as tseed

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", help="specify path to data dir.", default="../data")
parser.add_argument("-t", "--data_tag", help="tag specifying subgconfig of an experiment")
parser.add_argument("-r", "--rnd_seed", help="set random seed", default=0, type=int)

main_args = parser.parse_args()

tseed(main_args.rnd_seed)
npseed(main_args.rnd_seed)

# set hyper parameters
config = {}
classifier = "resnet"
train_size = 3_000_000
z_score_size = 100_000
z_score = True
config = {
    "model": classifier, 
    "z_score": z_score, 
    "train_size": train_size, 
    "z_score_size": z_score_size,
    "seed":main_args.rnd_seed,
}
train_kwargs = {
    "batch_size": 256, 
    "max_epochs": 5000, 
    "stop_after_epoch": 20, 
    "lr": 5e-4,
}
config.update(train_kwargs)

# init wandb
wandb.init(
    project="train_calibration_estimator",
    notes=(
        "Trains a calibration estimator to compensate for bias, caused by "
        "leaving out simulations too far from x_o and with nonsensical values."
        ), 
    config=config,
    settings=wandb.Settings(start_method='fork'),
)

data = SimpleDB(main_args.data_dir)

# import data
theta_train = data.query(f"theta_{main_args.data_tag}")
x_train = data.query(f"x_{main_args.data_tag}")
y_train = ~includes_nan(x_train) # True if no nans

# train estimator
bias_estimate = BiasEstimator(
    input_dim = theta_train.shape[1], 
    summarywriter = wandb, 
    model = classifier,
)
if z_score:
    bias_estimate.z_score_inputs(theta_train[:z_score_size])
bias_estimate.train(
    theta_train[:train_size],
    y_train[:train_size],
    verbose = True,
    **train_kwargs,
)
bias_estimate.summarywriter = None # remove wandb, since it cannot be pickled

data.write(f"calibration_estimator_{main_args.data_tag}_{main_args.rnd_seed}", bias_estimate)