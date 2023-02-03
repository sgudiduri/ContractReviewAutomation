from pathlib import Path
import typing as t
from pydantic import BaseModel, validator
from strictyaml import load, YAML
import os


# Project Directories
PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = Path(os.path.abspath(os.path.join(PWD, '..')))
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yaml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_model"
DATA_DIR = PACKAGE_ROOT / "data"

class AppConfig(BaseModel):
    """
    Application-level config.
    """
    package_name: str
    train_path: str
    test_path: str
    vocab_path: str
    model_path: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    num_step: int
    batch_size: int
    learning_rate: float
    embed_size: int
    num_hiddens: int
    epochs: int
    save_best: bool
    trainer: str
    loss: str

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
