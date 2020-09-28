"""
Manage experiments and config for PyTorch and Tensorboard
"""
import logging
from typing import Union, Sequence, Dict, List, Any, Optional
import argparse
from pathlib import Path
import random
import copy
import shutil
import yaml
import time
from os.path import expandvars
import numpy as np
import git
import torch


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    from tensorboardX import SummaryWriter


def recover_git_version(folder: Union[Path, str]) -> Dict[str, str]:
    repo = git.Repo(folder, search_parent_directories=True)
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
    return {"commit": repo.head.object.hexsha, "version": str(tags[-1])}


def read_config(folder_or_file: Union[Path, str]) -> Dict[str, Any]:
    """
    Read config from one file or from each yaml files contained on a folder.
    In the latter case, the files are read in the lexicographic order based on their names.
    """
    config: Dict[str, Any] = {}
    _folder_or_file = expand_path(folder_or_file)

    if _folder_or_file.is_file():
        return read_config_from_file(_folder_or_file)
    if _folder_or_file.is_dir():
        files = sorted([f for f in _folder_or_file.glob("*.y[a]ml")])
        for config_file in files:
            config = {**config, **read_config_from_file(config_file)}
        return config
    raise ValueError(f"Could not find {folder_or_file}")


def read_config_from_file(config_file: Union[str, Path]) -> Dict[str, Any]:
    with open(config_file, "r") as ymlfile:
        return yaml.safe_load(ymlfile)


def _clean_params(params: Dict, exclude: List[str] = [""]) -> Dict:
    """
    Produce a human-readable parameters file by removing abstractions
    """
    ret = {}
    for k, v in params.items():
        if k in exclude:
            continue
        if isinstance(v, Path):
            v = str(v)
        if isinstance(v, np.generic):
            v = v.tolist()
        if isinstance(v, tuple):
            v = list(v)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, torch.Tensor):
            v = v.tolist()
        if isinstance(v, Dict):
            v = _clean_params(v)
        ret[k] = v
    return ret


def load_params(name: str, all_params: Dict[str, Any]) -> Dict[str, Any]:
    if name not in all_params:
        raise ValueError(f"Can't find {name} in experiments")
    params: Dict[str, Any] = all_params[name]
    if "deps" in params:
        deps: Dict[str, Any] = {}
        for dep_name in params["deps"]:
            deps.update(**load_params(dep_name, all_params))
        params = {**deps, **params}
        del params["deps"]
    return params


class Exply:
    """
    Manage an experiment.
    """

    def __init__(
        self, name: str, data_folder: Union[Path, str] = "", **kwargs,
    ):
        """
        Parameters from config file, default, args, kwargs.

        Args:
        - config_yaml: folder of YAML files or 1 YAML file containing parameters
        - data_folder: folder on which should go every experiment
        """
        if name == "":
            raise ValueError("Please provide a meaningful name")

        parameters = {
            "name": name,
            "data_folder": data_folder,
            **kwargs,
        }

        # Create output_folder
        default_folder: Path = (
            Path(parameters["output_folder"])
            if "output_folder" in parameters
            else Path(parameters["data_folder"]) / parameters["name"]
        )
        self.build_output_folder(default_folder)
        self.output_folder = default_folder
        self.update(**parameters)
        logging.info(f"Data are saved into {self.output_folder}")

        self.checkpoint = self.output_folder / "checkpoint.pth"
        self.best_checkpoint = self.output_folder / "best_checkpoint.pth"
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.device = self.get("device", "cpu")

    @classmethod
    def from_yaml(cls, name: str, config_yaml: Union[Path, str]):
        configs = read_config(config_yaml)
        parameters = load_params(name, configs)
        return cls(name=name, **parameters)

    def get_tensorboard(self, name: str = "tb"):
        tb_path = expand_path(self.output_folder / name)
        tb_path.mkdir(exist_ok=True)
        return SummaryWriter(log_dir=tb_path)

    def post_epoch(self, epoch, metrics=None, **kwargs) -> bool:
        """
        Save the metrics, check early_stop and checkpointing
        """
        term = False

        # Early stopping
        if metrics is not None and self.get("best_metric_name") in metrics:
            term = self.early_stop(
                metrics[self.get("best_metric_name")], epoch=epoch + 1
            )

        # Storing metrics
        self.store_metrics(epoch, **metrics)

        # Checkpointing -- we let the manager deciding when to do it.
        self.checkpointing(epoch, metrics=metrics, **kwargs)

        return term

    def store_metrics(self, epoch, **metrics):
        """
        Save the metrics, hook them on TB
        """
        self.update(**metrics)

        if metrics is not None and self.get("best_metric_name") in metrics:
            self.update(best_metrics=metrics)

    def build_output_folder(self, folder: Union[str, Path], ignore_ext: Sequence = []):
        folder_path = expand_path(folder)
        ignore_exts = shutil.ignore_patterns(*ignore_ext)

        if not folder_path.is_dir():
            folder_path.mkdir(parents=True)
            folder_path.chmod(0o777)  # avoid permission denied on cluster

        if (folder_path / "parameters.yaml").is_file():
            self.import_parameters(folder_path / "parameters.yaml")

    def snapshot(
        self,
        folder: Union[str, Path],
        ignore_ext=[
            "*.pyc",
            "*__pycache__",
            "*egg-info",
            "*data",
            ".git",
            ".*",
            "tmp*",
            "tests",
            "demo",
        ],
    ) -> None:
        snapshots = self["output_folder"] / "snapshot"
        snapshots.mkdir(exist_ok=True)

        if not snapshots.is_dir():
            shutil.copytree(
                folder, snapshots, ignore=self.get("ignore_ext", ignore_ext)
            )

    def checkpointing(self, epoch, metrics=None, **kwargs):
        """
        Do the checkpointing and possibly backup it.
        """
        kwargs["epoch"] = epoch

        # Update the best checkpoint
        if metrics is not None and self["best_metric_name"] in metrics:
            if self.is_best_checkpoint(metrics[self.get("best_metric_name")], epoch):
                torch.save(kwargs, nb.expand_path(self["best_checkpoint"]))
        elif self.get("best_metric_name") != "":
            logging.warning(
                f"The metrics {self.get('best_metric_name')} is not available."
            )

        # Update the current checkpoint
        if (
            hasattr(self, "checkpoint_freq")
            and epoch % self.get("checkpoint_freq") == 0
        ):
            torch.save(kwargs, nb.expand_path(self.checkpoint))

        # Backup the checkpoint
        if (
            hasattr(self, "checkpoint_backup_freq")
            and epoch % self.get("checkpoint_backup_freq") == 0
        ):
            backup = nb.expand_path(
                self.checkpoint.parent
                / f"{self.checkpoint.stem}-{epoch}{self.checkpoint.suffix}"
            )
            torch.save(kwargs, backup)

    def is_best_checkpoint(self, loss: float, epoch: int) -> bool:
        """
        Save the checkpoint with the lowest loss
        """
        if self.best_loss is not None and self.best_loss < loss:
            return False

        self.best_loss = loss
        self.best_epoch = epoch
        return True

    def early_stop(self, loss: float, epoch: int) -> bool:
        """
        Stop training when its seems to have converged
        """
        # No early stopping
        if not hasattr(self, "patience") or self.get("patience") < 0:
            return False

        if self.is_best_checkpoint(loss, epoch):
            return False

        if self.best_epoch - epoch >= self.get("patience"):
            logging.info(f"Early stop: {self.best_loss}" f" @ {epoch}")
            return True

        return False

    def __repr__(self):
        ret = f"{self.__class__}\n"
        for k, v in self.__dict__.items():
            ret += f"\t- {k}: {v}\n"
        return ret

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __getitem__(self, name):
        if not hasattr(self, name):
            raise ValueError(f"The manager has not any parameter {name}")
        return self.get(name)

    def get(self, name, default=None):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return default

    def update(self, **kwargs):
        """Update several parameters at the same time

        >>> m = ExperimentManager(name='foo')
        >>> m.update(name='bar')
        >>> m.name
        'bar'
        """
        # We use super method to deal with user-defined setter
        for name, value in kwargs.items():
            super().__setattr__(name, value)

        self.export(self.output_folder / "parameters.yaml")

    def default_update(self, **kwargs):
        """Update several parameters at the same time but don't overwrite existing parameters

        >>> m = ExperimentManager(name='foo')
        >>> m.default_update(name='bar', subname='fi')
        >>> m.name
        'foo'
        >>> m.subname
        'fi'
        """
        # We use super method to deal with user-defined setter
        for name, value in kwargs.items():
            if not hasattr(self, name):
                super().__setattr__(name, value)

        self.export(self.output_folder / "parameters.yaml")

    def __setattr__(self, name, value):
        """
        Set an attribute

        >>> m = ExperimentManager(name='foo')
        Set an attribute

        'bar'
        """
        super().__setattr__(name, value)
        if name != "__dict__":
            self.export(self.output_folder / "parameters.yaml")

    def export(self, filename="", exclude=[]):
        params = _clean_params(self.__dict__, exclude)

        if filename != "":
            with open(expand_path(filename), "w") as outfile:
                yaml.dump(params, outfile, default_flow_style=False)

        return params

    def import_parameters(self, filename):

        with open(expand_path(filename), "r") as infile:
            params = yaml.load(infile, Loader=yaml.Loader)

        if params is None:
            logging.warning(f"Could not find parameters at {filename}")
            return

        if "output_folder" in params:
            params["output_folder"] = Path(params["output_folder"])

        self.default_update(**params)

    def requires(self, *keys: str):
        for key in keys:
            assert hasattr(self, key), f'Requires: {", ".join(keys)}. Cant get {key}.'

    def store_model(self, model: torch.nn.Module):
        with open(expand_path(self.output_folder) / "model.txt", "w") as fid:
            fid.write(str(model))

    def save_model(self, model: torch.nn.Module, optimizer=None, **kwargs):
        data = {
            "model_state_dict": model.state_dict(),
            **kwargs,
        }
        if optimizer is not None:
            data["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(
            data, expand_path(self.output_folder) / "checkpoint.pth",
        )

    def load_model(self, model: torch.nn.Module, optimizer=None):
        filename = expand_path(self.output_folder) / "checkpoint.pth"
        if not filename.is_file():
            raise ValueError(f"Checkpoint is not found at {filename}")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            print("Loading from epoch", checkpoint["epoch"])
        else:
            checkpoint["epoch"] = 0
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]


def expand_path(path: Union[str, Path]):
    return Path(expandvars(str(path)))


def add_metrics_tensorboard(
    epoch: int, metrics: Dict[str, float], tensorboard: SummaryWriter, manager: Exply,
):
    # Sort metrics per categories
    cats: Dict[str, Dict[str, float]] = {}
    for name, metric in metrics.items():
        parts: List[str] = name.split("_")
        cat_name: str = f"{parts[2]}_{parts[1]}" if len(parts) == 3 else name

        if cat_name not in cats:
            cats[cat_name] = {}

        cats[cat_name][name] = metric

    for name, cat in cats.items():
        tensorboard.add_scalars(f"{manager['name']}/{name}", cat, epoch)
