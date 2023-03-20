import json
import os
import pickle
import uuid
import warnings

# exceptions
from pickle import PicklingError
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from joblib import Parallel, delayed
from numpy import ndarray

# types
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from scipy.special import binom
from torch import Tensor
from tqdm.auto import tqdm

try:
    from ephys_helper.hh_simulator import HHSimulator as hhsim
except ModuleNotFoundError:
    print("ephys_helper not found. Required for HH simulation.")


def split_into_batches(collection: Tensor or List, batch_size: int) -> List or Tuple:
    if type(collection) == Tensor:
        batches = torch.split(collection, batch_size)
    if type(collection) == list:
        batches = [
            collection[i : i + batch_size]
            for i in range(0, len(collection), batch_size)
        ]

    return batches


def prepare4batched_execution(
    func: Callable, input_batch: List, batch_seed: Optional[int] = None
) -> List:
    """Encases function in a for loop in order to receive entire batch as input.
    Args:
        func: Function to be encased in for loop.
        input_batch: Batch of individual inputs to the function.
        batch_seed: Whether to seed the function with numpy and torch random seeds b4 execution.
    Returns:
        Batch of outputs."""

    output_cache = []

    if batch_seed != None:
        torch.manual_seed(batch_seed)
        np.random.seed(batch_seed)

    for input in input_batch:
        output = func(input)
        output_cache.append(output)

    return output_cache


def train_on_feature_subsets(
    training_fn: Callable,
    theta: Tensor,
    x: Tensor,
    subsets: List,
    batch_size: int = 1,
    n_workers: int = 1,
    batch_seed: int = None,
    **train_kwargs,
):
    """Enables to execute training on feature subsets in parallel.
    Args:
        training_fn: Function that takes training data {theta_n}_1:N,{x_n}_1:N
            and the feature indices as input and trains a density estimator on it.
            The trained estimator can either be supplied as output or saved,
            i.e. via a DataCollection.
        theta: Simulator parameters.
        x: Simulator outputs.
        subsets: List of lists that contain feature subsets, denoted by their index.
        batch_size: Number of tasks worked on per thread.
        n_workers: How many threads to use.
        batch_seed: Whether and with what value to seed each training process.
        train_kwargs: Can be used to supply additional kwargs, accepted by the
            training_fn.

        Returns:
            results: Results of the training process aka posterior estimators."""

    train = lambda feature_set: training_fn(theta, x, feature_set, **train_kwargs)
    train_batched = lambda feature_set_batch: prepare4batched_execution(
        train, feature_set_batch, batch_seed
    )

    results = hhsim.split_work(
        train_batched, subsets, batch_size=batch_size, n_workers=n_workers
    )
    return results


def sample_from_feature_subsets(
    sampling_fn: Callable,
    n_samples: int,
    context: Tensor,
    subsets: List,
    batch_size: int = 1,
    n_workers: int = 1,
    batch_seed: int = None,
    **sampling_kwargs,
):
    """Enables to execute training on feature subsets in parallel.
    Args:
        sampling_fn: Function that takes number of samples and context observation
            as input and draws samples from a density estimator. The drawn samples
            can either be supplied as output or saved, i.e. via a DataCollection.
        n_samples: Number of samples to draw per subset of features.
        context: Contextualises the posterior estimate.
        subsets: List of lists that contain feature subsets, denoted by their index.
        batch_size: Number of tasks worked on per thread.
        n_workers: How many threads to use.
        batch_seed: Whether and with what value to seed each training process.
        sampling_kwargs: Can be used to supply additional kwargs, accepted by the
            sampling_fn. This could include MCMC parameters.

        Returns:
            results: Results of the sampling processes."""

    sample = lambda feature_set: sampling_fn(
        n_samples, context, feature_set, **sampling_kwargs
    )
    sample_batched = lambda feature_set_batch: prepare4batched_execution(
        sample, feature_set_batch, batch_seed
    )

    results = hhsim.split_work(
        sample_batched, subsets, batch_size=batch_size, n_workers=n_workers
    )
    return results


def generate_random_feature_subsets(
    num_subsets: int, num_features: int, feature_list: List, seed: Optional[int] = None
) -> List:
    """Generate a number of subsets of features with a fixed size from a set of features.
    Args:
        num_subsets: Number of subsets to return.
        num_features: Size of the subsets to generate.
        feature_list: List of all features to draw from, for subsets.
        seed: Which seed to use. Enables reproducabillity.
    Returns:
        subsets: List of subsets of features.
    """
    if seed != None:
        np.random.seed(seed)
    subsets = []
    max_num = binom(len(subsets), num_features)
    while len(subsets) < num_subsets or len(subsets) < max_num:
        subset = list(np.random.choice(feature_list, num_features, replace=False))
        subset.sort()
        if subset not in subsets:
            subsets.append(subset)
    return subsets


class SimpleDB:
    r"""Simple interface for storing, tagging and files in a directory.

    Supports torch.Tensor, num, "r"py.ndarray, Dict, str, torch.distribution
    and all objects that are picklable.

    Objects can be written to cache or disc with `db.write(obj, tag)`.
    Stored objects can be retrieved with `db.query(tag)`. If an object is not
    cannot be located in cache, it will be read from disc. 

    Args:
        PATH: Location of stored files, when stored on disc.
        mode: Supports read 'r' and write 'w' mode. In read mode all functions
            that write are blocked.

    Attributes:
        location: Absolute path of directory on disc.
        items: Cached items with {tag: item}.
    """

    def __init__(self, PATH: str, mode="w"):
        self.location = os.path.abspath(PATH)
        self.items = {}
        self.init(mode)

    def __call__(self, key: str = None, all: bool = False) -> Dict or Any:
        """Retrieve a particular item or the whole database.

        If a key is provided, only that item will be returned.
        If no key is provided, either all cached or all stored items will be
        returned depending on `all`.

        Args:
            key: tag / unique identifier of an item.
            all: Whether to import items from disc, or just from cache.

        Returns:
            Queried item or all cached (and stored) items.
        """
        if key is None:
            if all:
                self.load_all()
            return self.items
        else:
            return self.query(key)

    def init(self, mode: str = "w"):
        """Creates directory @ PATH.
        
        Warns if directory already exists.
        
        Args: 
            mode: Supports read 'r' and write 'w' mode."""

        def read_only(*args, **kwargs):
            raise Exception("This operation is only supported in write mode.")

        if mode == "r":
            self.write2cache = read_only
            self.write2disc = read_only
            self.write = read_only

        elif mode == "w":
            try:
                os.mkdir(self.location)
            except FileExistsError:
                warnings.warn("Existing database opened in write mode. \
                    Risk to existing files. Consider opening it as read only.")
        else:
            raise ValueError("Mode not supported. Please selected 'w' or 'r'.")


    @staticmethod
    def is_picklable(obj: Any) -> bool:
        """Checks if obj is picklable.
        
        Tries to pickle object and checks for exceptions. If no exceptions are
        thrown the object is considered picklable.
        
        Args:
            obj: Any object.

        Returns:
            Wheter object is picklable."""
        try:
            pickle.dumps(obj)
            return True
        except PicklingError:
            return False

    @staticmethod
    def assert_legal_key(key: str):
        """Checks whether key contains only legal characters.

        Args:
            key: Unique identifier for item stored in `SimpleDB`.
        
        Raises:
            AssertionError: If key contains illegal characters.
        """
        assert isinstance(key, str), "Keys can only be strings"
        assert "." not in key, "Key cannot contain the character '.' "
        assert "/" not in key, "Key cannot contain the character '/' "

    def replace(self, key: str, item: Any, loc: str, warn: bool = True) -> str:
        """Reuses old key, if it already exists.

        This leads to items being overwritten.

        Args:
            key: Unique identifier for item stored in `SimpleDB`.
            item: Any object to be stored in the data base.
            loc: Whether to check on disc or in cache.
            warn: Warn if item is a duplicate.        
        """
        if loc == "cache":
            cond = self.item_in_cache(key)
        if loc == "disc":
            cond = self.item_on_disc(key)

        if cond and warn:
            warning = f"overwriting {key}"
            warnings.warn(warning)

        return key

    def rename(self, key: str, item: Any, loc: str, warn: bool = True) -> str:
        """Renames key, if it already exists.

        This leads to items being renamed.

        Args:
            key: Unique identifier for item stored in `SimpleDB`.
            item: Any object to be stored in the data base.
            loc: Whether to check on disc or in cache.
            warn: Warn if item is a duplicate.
        """
        if loc == "cache":
            cond = self.item_in_cache(key)
        if loc == "disc":
            cond = self.item_on_disc(key)

        if cond:
            if warn:
                warning = f"renaming {key}"
                warnings.warn(warning)
            return self.rename(key + "_copy", item, loc, warn=False)
        else:
            return key

    def find(self, string: str) -> Dict:
        """Search for items in the database, whose key matches a string.

        So far only supports looking whether a substring is contained within
        the item key.

        Args:
            string: Substring being matched to keys in the database.

        Returns:
            matching_items: Dictionary containing all the matching items.
        """
        # TODO: Add regex support
        cached_keys = [key for key in self.items]
        stored_items = os.listdir(self.location)
        stored_keys = [
            stored_key[: stored_key.find(".")] for stored_key in stored_items
        ]
        unique_keys = list(set(cached_keys + stored_keys))
        matching_keys = [key for key in unique_keys if string in key]
        matching_items = {}
        for key in matching_keys:
            matching_items[key] = self.query(key)
        return matching_items

    def item_on_disc(self, key: str) -> bool:
        """Check whether item exists on disc.
        
        Args:
            key: Unique identifier for item stored in `SimpleDB`.

        Returns:
            Whether key can be found on the disc.
        """
        stored_items = os.listdir(self.location)
        stored_keys = [
            stored_key[: stored_key.find(".")] for stored_key in stored_items
        ]
        return key in stored_keys

    def item_in_cache(self, key: str) -> bool:
        """Check whether item exists on disc.
        
        Args:
            key: Unique identifier for item stored in `SimpleDB`.

        Returns:
            Whether key can be found on the disc.
        """
        return key in self.items

    def write2cache(self, key: str, item: Any):
        """Add item to cache.

        Args:
            key: Unique identifier for item stored in `SimpleDB`.
            item: Any object to be stored in the data base.        
        """
        self.items[key] = item

    def write2disc(self, key: str, item: Any):
        """Write item to disc.
        
        Args:
            key: Unique identifier for item stored in `SimpleDB`.
            item: Any object to be stored in the data base.
        """
        # TODO: Warn if overwritten or renamed.
        if isinstance(item, dict):
            with open(f"{self.location}/{key}.json", "w") as f:
                f.write(json.dumps(item))
        elif isinstance(item, Tensor):
            torch.save(item, f"{self.location}/{key}.tpkl")
        elif isinstance(item, torch.distributions.Distribution):
            torch.save(item, f"{self.location}/{key}.tpkl")
        elif isinstance(item, ndarray):
            np.savez(f"{self.location}/{key}.npz", **{key: item})
        elif isinstance(item, str):
            with open(f"{self.location}/{key}.txt", "w") as f:
                f.write(item)
        elif isinstance(item, NeuralPosterior):
            torch.save(item, f"{self.location}/{key}.tpkl")
        elif self.is_picklable(item):
            with open(f"{self.location}/{key}.pkl","wb") as f:
                pickle.dump(item, f)
        else:
            raise ValueError("Item type is either not supported or picklable.")

    def write(
        self, key: str, item: Any, mode: str = "replace, disc, cache",
    ):
        """Store an item in the database.

        Depending on the mode that is selected the item is either written 
        (both can be selected):
        "disc" - to disc
        "cache" - to cache.
        The way existing items are handled can also be specified. Two strategies
        are implemented (only one can be selected):
        "replace" - overwrites existing items
        "rename" - renames the item being written to "key_copy"

        Args:
            key: Unique identifier for item stored in `SimpleDB`.
            item: Any object to be stored in the data base.
            mode: How to write item to the database and how to handle duplicates.
        
        Raises:
            ValueError: If no duplicate strat is selected.
            ValueError: If item is not picklable or supported.
        """
        self.assert_legal_key(key)

        if "cache" in mode.lower():
            if "replace" in mode.lower():
                new_key = self.replace(key, item, "cache")
            elif "rename" in mode.lower():
                new_key = self.rename(key, item, "cache")
            else:
                raise ValueError("No available method was selected for writing.")
            self.write2cache(new_key, item)

        if "disc" in mode.lower():
            if "replace" in mode.lower():
                new_key = self.replace(key, item, "disc")
            elif "rename" in mode.lower():
                new_key = self.rename(key, item, "disc")
            else:
                raise ValueError("No available method was selected for writing.")
            self.write2disc(new_key, item)

    def query(self, key: str) -> Any:
        """Query database for items.

        First the cache will be queried, when the item exists in cache it will be
        returned. If it does not exist in cache, it will be retrieved from disc.

        Args:
            key: Unique identifier for item stored in `SimpleDB`.

        Returns:
            Queried item, if it has a matching tag.
        
        Raises:
            KeyError: If key does not match item in the database.
        """
        self.assert_legal_key(key)
        try:
            return self.items[key]
        except KeyError:
            return self.load_item(key)

    def load_all(self, add2cache=True) -> Dict:
        """Load all items currently stored on disc into cache.

        Args:
            add2cache: Whether to add item to `self.items`.

        Returns:
            All items that are contained in the cache after this operation.
        """
        stored_files = os.listdir(self.location)
        supported_formats = [".pkl", ".tpkl", ".txt", ".log", ".json", ".npz"]
        supported_files = [
            file for file in stored_files if file[file.find(".") :] in supported_formats
        ]
        stored_keys = [key[: key.find(".")] for key in supported_files]

        all_items = {}
        for key in stored_keys:
            all_items.update({key: self.load_item(key, add2cache=add2cache)})

        all_items.update(self.items)
        return all_items

    def load_item(self, key: str, add2cache: bool = False) -> Any:
        """Load item from disc.

        Query items on disc. If the filetype is supported, the item will be 
        returned as one of the supported output types (torch.Tensor, 
        numpy.ndarray, str, Dict, torch.Distribution or some unpickled object).

        If add2cache is selected, the item will be added to the item cache.
        
        Args:
            key: Unique identifier for item stored in `SimpleDB`.
            key: Unique identifier for item stored in `SimpleDB`.
            add2cache: Whether to add item to `self.items`.

        Returns:
            queried item.

        Raises:
            TypeError: If item file type is not supported.
        """
        self.assert_legal_key(key)
        try:
            stored_fnames = os.listdir(self.location)
            matching_files = [
                stored_fname
                for stored_fname in stored_fnames
                if f"{key}" == stored_fname.split(".")[0]
            ]
            assert len(matching_files) <= 1, "More than one matching key found"
            item_file = matching_files[0]
            filetype_idx = matching_files[0].find(".") + 1
            filetype = matching_files[0][filetype_idx:]
            if filetype == "pkl":
                with open(f"{self.location}/{item_file}", "rb") as f:
                    item = pickle.load(f)
            elif filetype == "tpkl":
                item = torch.load(f"{self.location}/{item_file}")
            elif filetype == "npz":
                with np.load(f"{self.location}/{item_file}") as data:
                    item = data[key]
            elif filetype == "txt":
                with open(f"{self.location}/{item_file}", "r") as f:
                    item = f.read()
            elif filetype == "log":
                with open(f"{self.location}/{item_file}", "r") as f:
                    item = f.read()
            elif filetype == "json":
                with open(f"{self.location}/{item_file}", "r") as f:
                    item = json.load(f)
            else:
                raise TypeError("Filetype not supported")

            if add2cache:
                self.write2cache(key, item)
            return item
        except (IndexError, KeyError):
            raise KeyError("Key does not match item in the database.")

    def info(self, key: str = None, show_names: bool = False):
        """Prints some useful information about an item or the database.

        Args:
            key: Unique identifier for item stored in `SimpleDB`.
            show_names: Whether to print file names of all items in database.  
        """
        print(f"The database is located @ {self.location}")

        metadata = self.find("metadata")
        if metadata != {}:
            for key, val in metadata.items():
                print(f"{key} - {val}.")

        if key is None:
            stored_items = os.listdir(self.location)
            if show_names:
                print(stored_items)
            all_items = self.load_all(False)
            print(f"{len(all_items)} unique key are currently stored in the database")
            print(f"{len(self.items)} items are currently cached.")
            print(f"{len(stored_items)} items are currently stored.")
        else:
            if self.item_in_cache(key):
                cached_item = self.load_item(key)
                print(f"{key} [{type(cached_item)}] is cached.")
            if self.item_on_disc(key):
                stored_item = self.items[key]
                print(f"{key} [{type(stored_item)}] is on disc.")
            else:
                print("The key does not match an item in the database.")

    def configure(self):
        print("This function does nothing at the moment")
        # possible config stuff
        pass


class Task:
    """Routine that runs on a single thread and can be parallelised.

    Supports callback and error callback functionality.

    Can be used in conjunction with TaskMnanager to parallelise tasks.

    Args:
        task: Routine that gets run.
        args: Arguments that the routine takes.
        kwargs: Keywordarguments that the routine takes.
        callback: Function that gets called upon task completion. Takes the 
            object that is returned by task.
        callback_kwargs: Keyword arguments for callback.
        error_callback: Function that gets called if task raises error. Takes
            the error as input.
        name: Task name, to identify it.
        priority: If the task is queued, higher priority tasks are executed first.
        store_results: Whether to store the result of the calculation in the
            result attribute.

    Attributes:
        _task: Routine that gets run.
        args: Arguments that the routine takes.
        kwargs: Keywordarguments that the routine takes.
        _callback: Function that gets called upon task completion. Takes the 
            object that is returned by task.
        callback_kwargs: Keyword arguments for callback.
        _error_callback: Function that gets called if task raises error. Takes
            the error as input.
        name: Task name, to identify it.
        priority: If the task is queued, higher priority tasks are executed first.
        _store_results: Whether to store the result of the calculation in the
            result attribute.
        id: UUID that uniquely identifies a given task.
        result: Stores the result of the task after it was run and if store_result
            was selected.
    """

    def __init__(
        self,
        task: Callable,
        args: Optional[List] = (),
        kwargs: Optional[Dict] = {},
        callback: Optional[Callable] = None,
        callback_kwargs: Optional[Dict] = {},
        error_callback: Optional[Callable] = None,
        name: Optional[str] = None,
        priority: int = 1,
        store_result=False,
    ):
        self._task = task
        self.args = args
        self.kwargs = kwargs
        self._callback = callback
        self.callback_kwargs = callback_kwargs
        self._error_callback = error_callback
        self.name = name
        self.priority = priority
        self._store_result = store_result

        self.id = uuid.uuid4().hex
        self.result = None

    def __call__(self, *args, **kwargs) -> Any:
        r"""Provides passthrough to `self.executre()`
        """
        return self.execute(*args, **kwargs)

    def store_result(self, val=True):
        """Set self.store_result.
        
        Args:
            val: Whether to store result or not.
        """
        self._store_result = val

    def execute(self, *args, **kwargs) -> Any:
        r"""Runs the task.
        If args and kwargs are provided, they are used. Else `self.args` and 
        `self.kwargs` are used.

        Returns:
            `task(*args, **kwargs)`

        Raises:
            TypeError: If arguments are missing.
        """
        if args != () or kwargs != {}:
            result = self._task(*args, **kwargs)
        else:
            try:
                result = self._task(*self.args, **self.kwargs)
            except TypeError:
                raise TypeError("Necessary arguments have not been provided.")

        if self._store_result:
            self.result = result
        return result

    def callback(self, task_output: Any) -> Any:
        r"""Provides customisable callback funcitonality.

        Args:
            task_output: Takes the output of `task()` as input.

        Returns:
            `callback(task_output, **callback_kwargs)`
        """
        if self._store_result:
            self.result = task_output
        if self._callback is not None:
            return self._callback(task_output, **self.callback_kwargs)

    def error_callback(self, task_error: Exception) -> Any:
        """If task throws error, it can be handled by the provided error callback.
        
        Args:
            task_error: An exception that is thrown from calling `task()`
            
        Returns:
            `error_callback(task_error)`
        """
        if self._error_callback is not None:
            return self._error_callback(task_error)


class TaskManager:
    r"""Handles the prioritisation, distribution and execution of `Task`s.

    Example:
    ```
    task1 = Task(func1)
    task2 = Task(func2)
    queue = TaskManager([task1, task2])
    queue.execute_tasks()
    ```

    Args:
        tasks: A list of Tasks.
        max_workers: How many workers to distribute the tasks to.

    Attributes:
        tasks: A list of Tasks.
        max_workers: How many workers to distribute the tasks to.
    """

    def __init__(self, tasks, max_workers=-1):
        self.tasks = tasks
        self.max_workers = max_workers

    def store_results(self, val=True):
        """Set self.store_result of individual tasks.
        
        Args:
            val: Whether to store result or not.
        """
        # if isinstance(self.tasks, TaskBundle):
        #     self.tasks.store_results(val)
        # else:
        for task in self.tasks:
            task.store_result(val)

    @property
    def max_workers(self):
        return self._max_workers

    @max_workers.setter
    def max_workers(self, val):
        """Set max number of workers.
        
        If set to -1, it will use all available workers.
        
        Args:
            val: number of workers.    
        """
        assert (val != 0) and (val >= -1), "specify a valid number of workers."
        if val == -1 or val == "all":
            val = mp.cpu_count()
        self._max_workers = val

    def apply_tasks2pool(self, task_queue: List[Task]) -> List:
        """Apply task list to pool of workers.
        
        Args:
            task_queue: List of tasks.
            
        Returns:
            results: List of results for each task in order of tasks."""
        with Parallel(n_jobs=self.max_workers) as parallel:
            results = parallel(
                delayed(task)()
                for task in tqdm(
                    task_queue,
                    disable=False,
                    desc=f"Running {len(task_queue)} tasks on {self.max_workers} threads",
                    total=len(task_queue),
                )
            )
        # TODO: Find out why apply_async stalls or sometimes does not work!

        # with Pool(processes=self.max_workers) as pool:
        #     outputs = [pool.apply_async(task.execute, task.args, task.kwargs, callback=task.callback, error_callback=task.error_callback) for task in task_queue]
        #     results = [res.get() for res in outputs]
        # print("closed pool")
        return results

    def imap_tasks2pool(
        self,
        task: Callable,
        iterable: Generator,
        callback: Callable = None,
        callback_kwargs: Dict = {},
    ) -> List:
        """Map task and itterable to worker pool.

        Args:
            task: Callable that takes args and kwargs from iterable.
            iterable: That contains args and kwargs for task.
            callback: Gets called upon completion of the task.
            callback_kwargs: Kwargs of callback.

        Returns:
            results: List of results for each tasks in order of tasks.
        """
        with Parallel(n_jobs=self.max_workers) as parallel:
            results = parallel(
                delayed(task)(*args, **kwargs) for args, kwargs in iterable
            )

        # TODO: Find out why imap_async stalls or sometimes does not work!

        # results = []
        # with Pool(processes=self.max_workers) as pool:
        #     if callback is not None:
        #         for result in zip(pool.imap(task, iterable), callback, callback_kwargs):
        #             results.append([result])
        #             callback(result, **callback_kwargs)
        #     else:
        #         for result in pool.imap_unordered(task, iterable):
        #             results.append([result])
        # print("closed pool")
        return results

    def execute_tasks(
        self, seperate_priorities: bool = True, store_results: bool = False
    ) -> List:
        """Starts the taskmanager. Distributes tasks over multiple processes.

        Args:
            seperate_priorities: Whether to seperate out the excecution of tasks
                grouped by their priorities. That means no task of lower priority
                gets started until all tasks of lower priority have finished.
            store_results: Whether to store the result of each computation in 
                task.        
        """
        self.store_results(store_results)

        # # if Task Bundle
        # if isinstance(self.tasks, TaskBundle):
        #     try:
        #         # if func + itterable
        #         func, gen = self.tasks.get_task_map()
        #         callback, callback_kwargs = self.tasks.get_callback_map()
        #         results = self.imap_tasks2pool(func, gen, callback, callback_kwargs)
        #     except AssertionError:
        #         # if TaskBundle from task list
        #         self.tasks = self.tasks()

        # if Task list
        if isinstance(self.tasks, List) or isinstance(self.tasks, Generator):
            results = []
            priorities = sorted(
                list(set([task.priority for task in self.tasks])), reverse=True
            )
            if seperate_priorities:
                for prio in priorities:
                    queue = [task for task in self.tasks if task.priority == prio]
                    results.append(self.apply_tasks2pool(queue))
            else:
                tasks_in_order = sorted(self.tasks, key=lambda x: x.priority)
                results = self.apply_tasks2pool(tasks_in_order)
        return results

    def results(self) -> Dict:
        """Accumulates all the results stored in the individual tasks.
        
        Returns:
            Dictionary with task name and task output."""
        return {task.name: task.result for task in self.tasks}


def str2int(string: str, n_digits: int = 9) -> int:
    """Turns string into integer.

    Can be used for seeding purposes.

    Args:
        string: String.
        n_digits: Takes only last N digits of the integer.
    """
    str_as_int = int.from_bytes(string.encode(), "little")
    sliced_int = int(str(str_as_int)[-n_digits:])
    return sliced_int
