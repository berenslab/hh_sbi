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
from fslm.metrics import mmd, sample_kl
from fslm.snle import ReducablePosterior
from fslm.utils import record_scaler
import wandb

from fslm4expdata.utils import import_tree_data
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------
# init experiment

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rnd_seed", help="set random seed", default=0, type=int)
parser.add_argument(
    "-m",
    "--method",
    help="set method to select feature subsets. 'best' or 'random' ",
    default="best",
)
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
    "-w", "--workers", help="how many workers to use.", default=-1, type=int
)
parser.add_argument(
    "-d", "--data_dir", help="from which directory to import presimulated the data"
)
parser.add_argument(
    "-p", "--posterior_dir", help="from which directory to import pretrained posterior"
)
parser.add_argument("-t", "--tag_of_posterior", help="specify which posterior by tag")
parser.add_argument(
    "-f",
    "--root_folder",
    help="specify root folder for data, experiment and posterior dir.",
    default="./",
)
parser.add_argument(
    "-o", "--observation_name", help="name of observation"
)
parser.add_argument("-F", "--features", help="list of features", default="list(range(23))")


main_args = parser.parse_args()

# Set random seed
npseed(main_args.rnd_seed)
tseed(main_args.rnd_seed)

MAX_DEPTH = 4

# init database and logger
db = SimpleDB(main_args.root_folder + main_args.name)
data = SimpleDB(main_args.root_folder + main_args.data_dir)
posteriors = SimpleDB(main_args.root_folder + main_args.posterior_dir)

wandb.init(project="hh_tree", entity="jnsbck", config=main_args, settings=wandb.Settings(start_method='fork'))

# -------------------------------------------------------------------------------
# import prior and observations
X_o = data.query("X_o")
x_o = torch.tensor([X_o[f"{main_args.observation_name}"]])[:,:-4]

prior = data.query("prior")
wandb.log({"observation": x_o})

# select feature indices to use
all_dims = eval(main_args.features)
wandb.config.update({"dims": all_dims})

# -------------------------------------------------------------------------------
def sampling_loop(full_posterior, n_samples, context, dims, method_tag, **kwargs):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"{method_tag}_{dims}") + main_args.rnd_seed
    npseed(seed)
    tseed(seed)

    posterior = ReducablePosterior(full_posterior)
    posterior.marginalise(dims)

    try:
        thetas = db.query(f"{method_tag}_{dims}")
    except KeyError:
        thetas = posterior.sample((n_samples,), context, **kwargs)
        db.write(f"{method_tag}_{dims}", thetas, mode="replace, disc")

    return thetas


def sample_and_test(X, ft, *args, **kwargs):
    Y = sampling_loop(*args, **kwargs)
    return ft, sample_kl(X, Y)
    # return ft, mmd(X,Y)


# -------------------------------------------------------------------------------
# define recursive tree search
def tree_search(full_posterior, remaining_fts=None, good_fts=[], div_fts=[], depth=0):
    if remaining_fts is None:
        remaining_fts = all_dims

    if depth == 0:
        # Sample full posterior
        args = (full_posterior, 1000, x_o, all_dims, method_tag)
        kwargs = {"warmup_steps": 100, "num_workers":4} # with thinning factor of 10 -> 4 chains ensures each is at least 2.5k steps long
        task = Task(
            task=sampling_loop,
            args=args,
            kwargs=kwargs,
            name=f"{method_tag}_{all_dims}",
            priority=2,
        )
        task.execute()
        X = db.query(f"{method_tag}_{all_dims}")
        remaining_fts = all_dims
    elif depth >= MAX_DEPTH+1:
        print("Maximum recursion depth exceeded.")
        return good_fts, div_fts # BREAK RECURSION IF DEPTH EXCEEDS LIMIT
    else:
        X = db.query(f"{method_tag}_{all_dims}")

    task_queue = []
    for ft in remaining_fts:
        dims = sorted(good_fts.copy() + [ft])
        args = (X, ft, full_posterior, 1000, x_o, dims, method_tag)
        kwargs = {"warmup_steps": 100}
        task = Task(
            task=sample_and_test,
            args=args,
            kwargs=kwargs,
            name=f"{method_tag}_{dims}",
            priority=2,
        )
        task_queue.append(task)

    mp_queue = TaskManager(task_queue, main_args.workers)
    out = torch.tensor(mp_queue.execute_tasks()[0])
    assert not torch.any(abs(out).isnan()) and not torch.any(
        abs(out).isinf()
    ), "Metric contains NaNs or Infs."

    divs = {int(key): float(val) for key, val in out[out[:, 0].sort()[1]]}

    # choose best feature
    if main_args.method == "best":
        next_best_ft = min(divs, key=divs.get)
    # choose random feature
    elif main_args.method == "random":
        next_idx = torch.randperm(len(remaining_fts))[0] # select random index
        next_best_ft = remaining_fts[next_idx]
    else:
        raise ValueError("Provide featuresets kwarg. Either 'best' or 'random' ")

    good_fts.append(next_best_ft)
    next_best_div = divs[next_best_ft]
    div_fts.append(next_best_div)
    record_scaler(
        next_best_div, next_best_ft, db.location + f"/{method_tag}_best_fts.txt"
    )
    wandb.log({"ft": next_best_ft, "div": next_best_div})

    remaining_fts = [d for d in all_dims if d not in good_fts]
    
    p = {key:float("nan") for key in all_dims}
    p = {key:(f"{divs[key]:.3f}" if key in divs else f"{val:.3f}") for key,val in p.items()}
    p_str =f"depth = {depth}: {', '.join(p.values())}"
    
    if depth >= MAX_DEPTH or len(remaining_fts) == 0:
        print(len(p_str)*"-")
        return
    else:
        print(p_str)
        tree_search(full_posterior, remaining_fts, good_fts, div_fts, depth + 1)
        return good_fts, div_fts

def try2resume_treesearch(db, method_tag, dims = list(range(23))):

    name = list(db.find(f"{method_tag}_best_fts").keys())
    assert len(name) <= 1, "More than one file found to resume from."
    
    try:
        assert len(name) != 0, "No file found to resume from."
        prev_dict = import_tree_data(f"{db.location}/{name[0]}.txt")
        prev_selected_fts = list(prev_dict.keys())
        prev_selected_divs = list(prev_dict.values())
        remaining_fts = [ft for ft in dims if ft not in prev_selected_fts]
        depth = len(dims)-len(remaining_fts)
        print(f"resuming from depth = {depth} with remaining fts: {remaining_fts}")
        return remaining_fts, prev_selected_fts, prev_selected_divs, depth
    
    except AssertionError:
        print("No resumable state found. Starting from beginning")
        return all_dims, [], [], 0

# -------------------------------------------------------------------------------
# import posterior
method_tag = f"{main_args.method}_{main_args.sample_with}_{main_args.rnd_seed}"
full_posterior_dict = posteriors.find(main_args.tag_of_posterior)
full_posterior_dict = {
    key: val for key, val in full_posterior_dict.items() if "posterior" in key
}
if len(full_posterior_dict) == 1:
    key, full_posterior = full_posterior_dict.popitem()
    db.write(
        f"posterior_loc_{method_tag}",
        f"the location of the posterior used: {posteriors.location}/{key}",
        mode="replace, disc",
    )
else:
    raise ValueError(
        "The tag does not uniquely identify one posterior. Please be more specific"
    )

# -------------------------------------------------------------------------------
# run tree search
remaining_fts, selected_fts, selected_divs, depth = try2resume_treesearch(db, method_tag, all_dims)
good_fts, divs = tree_search(full_posterior, remaining_fts, selected_fts, selected_divs, depth)
print(good_fts, divs)
# # -------------------------------------------------------------------------------
# # plot features
# fig, ax = plt.subplots(figsize=(4.7,2.1))
# ax.plot(range(len(all_dims)), divs, ".-", c="black", ms=6)
# ax.set_ylabel(r"$\mathcal{D}_{KL}[\hat{p}\vert\vert \hat{p}_{\backslash i}]$")
# ax.set_xlabel(r"Selected features")

# ax.set_xticks(range(len(all_dims)))
# ax.set_xticklabels(good_fts, rotation=90, color="black")

# wandb.log({"best_fts": fig})