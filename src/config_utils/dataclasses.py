@dataclass
class Paths:
    log: str
    data: str
    results: str


@dataclass
class Files:
    data: str


@dataclass
class DataLabels:
    features: list[str]
    response: list[str]


@dataclass
class Params:
    pass


@dataclass
class ContinuousParams(Params):
    search: str
    type: str
    range: list[int | float]


@dataclass
class DiscreteParams(Params):
    ...


@dataclass
class StaticParams(Params):
    ...


@dataclass
class BoolParams(Params):
    values: list[str]


@dataclass
class ModelConfig:
    model_type: str
    hyperparamers: Params


@dataclass
class MLPConfig(ModelConfig):
    type: str


@dataclass
class MLRConfig(ModelConfig):
    values: list[str]


@dataclass
class RFConfig(ModelConfig):
    type: str


@dataclass
class SVMConfig(ModelConfig):
    ...


@dataclass
class CaseStudyConfig:
    paths: Paths
    files: Files
    data_labels: DataLabels
    model: ModelConfig
