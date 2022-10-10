from __future__ import annotations
from .custom_task import CustomMeta

import importlib_resources
from omegaconf import OmegaConf

def _custom_resource_file_path(fname) -> str:
    with importlib_resources.path("minedojo.tasks.custom_tasks", fname) as p:
        return str(p)

CUSTOM_TASKS = OmegaConf.load(_custom_resource_file_path("custom_tasks_specs.yaml"))