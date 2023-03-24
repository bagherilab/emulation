from dataclasses import dataclass


@dataclass
class Paths:
    """Paths to the data and results directories."""

    log: str
    data: str
    results: str


@dataclass
class Files:
    """File names for the data and results files."""

    data: str


@dataclass
class DataLabels:
    """Labels for the data and results files."""

    features: list[str]
    response: list[str]


@dataclass
class Params:
    """Base class for hyperparameters."""

    pass


@dataclass
class ContinuousParams(Params):
    """Continuous hyperparameters."""

    search: str
    type: str
    range: list[int | float]


@dataclass
class DiscreteParams(Params):
    """Discrete hyperparameters."""

    ...


@dataclass
class StaticParams(Params):
    """Static hyperparameters."""

    ...


@dataclass
class BoolParams(Params):
    """Boolean hyperparameters."""

    values: list[str]


@dataclass
class ModelConfig:
    """Base class for model configurations."""

    model_type: str
    hyperparamers: Params


@dataclass
class MLPConfig(ModelConfig):
    """Multi-layer perceptron configuration."""

    type: str


@dataclass
class MLRConfig(ModelConfig):
    """Multiple linear regression configuration."""

    values: list[str]


@dataclass
class RFConfig(ModelConfig):
    """Random forest configuration."""

    type: str


@dataclass
class SVMConfig(ModelConfig):
    """Support vector machine configuration."""

    ...


@dataclass
class CaseStudyConfig:
    """Class for case study configurations."""

    paths: Paths
    files: Files
    data_labels: DataLabels
    model: ModelConfig
