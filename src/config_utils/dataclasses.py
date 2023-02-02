from dataclasses import dataclass
from typing import Optional

""" Experiment config """
@dataclass
class Experiment:
    name: str

@dataclass
class Files:
    data: str

@dataclass
class Paths:
    log: str
    data: str
    results: str

@dataclass
class Data:
    features: list[str]
    response: list[str]


@dataclass
class Params:
    pass

""" Model config """
@dataclass
class ContinuousParams(Params):
    type: str
    range: list[int | float]
    search: str
    

@dataclass
class DiscreteParams(Params):
    activation: Optional[list[str]]
    hidden_layer_size: Optional[list[str]]
    bootstrap: Optional[list[bool]]
    kernel: Optional[list[str]]


@dataclass
class StaticParams(Params):
    solver: Optional[str]
    max_iter: Optional[int]


@dataclass
class HyperParams:
    continous: Optional[ContinuousParams]
    discrete: Optional[DiscreteParams]
    static: Optional[StaticParams]


@dataclass
class ModelConfig:
    hyperparameters: HyperParams
    

@dataclass
class ExperimentConfig:
    models: dict[str, ModelConfig]
    experiment: Experiment
    files: Files 
    paths: Paths 
    data: Data 


@dataclass 
class CaseStudyConfig:
    cs: ExperimentConfig
    sobol_power: int 
    debug: Optional[bool] = False