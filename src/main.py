import hydra
from hydra.core.config_store import ConfigStore

from config import CaseStudyConfig

cs = ConfigStore.instance()
cs.store(name="case_study", node=CaseStudyConfig)  # loads according to type hinting in config.py
# Case Study 1
def finley_case_study():
    raise NotImplementedError()


# Case Study 2
def tumorcode_case_study():
    raise NotImplementedError()


# Case Study 3


def arcade_case_study():
    raise NotImplementedError()


@hydra.main(config_path="conf", config_name="config")
def main():
    # finley_case_study()
    tumorcode_case_study()
    # arcade_case_study()


if __name__ == "__main__":
    main()
