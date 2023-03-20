import argparse
import torch

from numpy.random import seed as npseed
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
    "-n", "--name", help="set name of the experiment.", default="default_experiment"
)
parser.add_argument(
    "-s",
    "--sample_with",
    help="whether to sample with mcmc or rejection.",
    default="mcmc",
)
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
# parser.add_argument(
#     "-o", "--observation_name", help="name of observation"
# )
parser.add_argument(
    "-o", "--observation_index", help="index of observation", type=int
)
parser.add_argument(
    "-t", "--data_tag", help="tag specifying subgconfig of an experiment"
)

main_args = parser.parse_args()

# Set random seed
npseed(main_args.rnd_seed)
tseed(main_args.rnd_seed)

# init database and logger
db = SimpleDB(main_args.root_folder + main_args.name)
data = SimpleDB(main_args.root_folder + main_args.data_dir)

wandb.init(
    name=f"{main_args.name}: xo={main_args.observation_index}, seed={main_args.rnd_seed}",
    project="sample_hh_nle", 
    entity="jnsbck", 
    config=main_args, 
    settings=wandb.Settings(start_method='fork'),
)

# -------------------------------------------------------------------------------
# import prior and observations
X_o = data.query("X_o")
# x_o = torch.tensor([X_o[f"{main_args.observation_name}"]])[:,:-4]
observation_name, x_o = list(X_o.items())[main_args.observation_index]
x_o = torch.tensor([x_o])[:,:-4]

prior = data.query("prior")
wandb.log({"observation": x_o})

# select feature indices to use
all_dims = list(range(x_o.shape[1]))
wandb.config.update({"dims": all_dims})

# -------------------------------------------------------------------------------
# define sampling loop
def sampling_loop(n_samples, context, dims, method_tag, **kwargs):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"samples_{method_tag}_{dims}") + main_args.rnd_seed
    npseed(seed)
    tseed(seed)
    posterior = data.query(f"posterior_{method_tag}_full")
    context = context[:, dims].view(1, -1)

    thetas = posterior.sample((n_samples,), context, **kwargs)

    # db.write(f"samples_{method_tag}_{main_args.observation_name}_all_dims", thetas)
    db.write(f"samples_{method_tag}_{observation_name}_all_dims", thetas)
    return thetas

# -------------------------------------------------------------------------------
# create tasks for training and sampling and add them to task queue
method_tag = f"nle_{main_args.data_tag}_{main_args.sample_with}_{main_args.rnd_seed}"

task_queue = []

num_samples = 1000
args = (num_samples, x_o, all_dims, method_tag)
kwargs = {"warmup_steps": 100}
task = Task(
    task=sampling_loop,
    args=args,
    kwargs=kwargs,
    name=f"sample_{method_tag}_{all_dims}",
    priority=2,
)
# task_queue.append(task)
wandb.config.update({"sampling_kwargs": kwargs})
wandb.config.update({"num_samples": num_samples})
sampling_loop(*args, **kwargs)
# -------------------------------------------------------------------------------
# dispatch tasks to task handler
# mp_queue = TaskManager(task_queue, main_args.workers)
# mp_queue.execute_tasks()
