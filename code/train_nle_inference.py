import argparse
import torch

from numpy.random import seed as npseed
from sbi.inference.snle import SNLE_A
from torch import manual_seed as tseed

from fslm.experiment_helper import (
    SimpleDB,
    Task,
    TaskManager,
    str2int,
)
import wandb

# -------------------------------------------------------------------------------
# init experiment

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rnd_seed", help="set random seed", default=0, type=int)
parser.add_argument(
    "-w",
    "--workers",
    help="how many workers are used in parallel.",
    default=-1,
    type=int,
)
parser.add_argument(
    "-d", "--data_dir", help="from which directory to import the presimulated data"
)
parser.add_argument(
    "-f",
    "--root_folder",
    help="specify root folder for data and experiment dir.",
    default="./",
)
parser.add_argument(
    "-t", "--data_tag", help="tag specifying subgconfig of an experiment"
)
parser.add_argument(
    "-b", 
    "--batchsize", 
    help="number of training pairs per batch.",
    default=100, 
    type=int,
)
parser.add_argument(
    "-l", 
    "--lr", 
    help="learning rate",
    default=0.0005, 
    type=float,
)

main_args = parser.parse_args()
config = {}

# Set random seed
npseed(main_args.rnd_seed)
tseed(main_args.rnd_seed)

# init database
data = SimpleDB(main_args.root_folder + main_args.data_dir)

# -------------------------------------------------------------------------------
# import prior and observations
prior = data.query("prior")

x = data.query(f"x_{main_args.data_tag}")
theta = data.query(f"theta_{main_args.data_tag}")

# select feature indices to use
all_dims = list(range(x.shape[1]))
config.update({"dims": all_dims})

# -------------------------------------------------------------------------------
# define training loop
def training_loop(theta, x, dims, prior, method_tag, lr, summary_writer):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"posterior_{method_tag}_{dims}") + main_args.rnd_seed
    npseed(seed)
    tseed(seed)

    inference = SNLE_A(
        prior, 
        show_progress_bars=True, 
        density_estimator="mdn", 
        summary_writer=summary_writer,
    )

    inference.append_simulations(theta, x[:, dims], exclude_invalid_x=True).train(
        training_batch_size=100, 
        learning_rate=lr,
    )

    data.write(f"inference_{method_tag}_full", inference)
    return inference

# -------------------------------------------------------------------------------
# create tasks for training and sampling and add them to task queue
method_tag = f"nle_{main_args.data_tag}_{main_args.batchsize}_{main_args.lr:.0e}_{main_args.rnd_seed}"

kwargs = {}
args = (theta, x, all_dims, prior, method_tag, main_args.lr)

config.update(vars(main_args))
config.update({"training_kwargs": kwargs})

# init logger
wandb.init(
    project="train_hh_nle", 
    entity="jnsbck", 
    config=config, 
    sync_tensorboard=True, 
    settings=wandb.Settings(start_method='fork'),
)
writer = torch.utils.tensorboard.SummaryWriter()
args += (writer,)

# task_queue = []
# task = Task(
#     task=training_loop,
#     args=args,
#     kwargs=kwargs,
#     name=f"train_{method_tag}_{all_dims}",
#     priority=4,
# )
# task_queue.append(task)

training_loop(*args, **kwargs)

# -------------------------------------------------------------------------------
# dispatch tasks to task handler
# mp_queue = TaskManager(task_queue, main_args.workers)
# mp_queue.execute_tasks()
