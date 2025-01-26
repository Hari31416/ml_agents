from .env import LOG_LEVEL
import numpy as np
from typing import Union, Any
import logging
import os


A = np.ndarray
END = "\033[0m"
BOLD = "\033[1m"
BROWN = "\033[0;33m"
ITALIC = "\033[3m"


def set_logger_level_to_all_local(level: int) -> None:
    """Sets the level of all local loggers to the given level.

    Parameters
    ----------
    level : int, optional
        The level to set the loggers to, by default logging.DEBUG.
    """
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]

    for _, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            if hasattr(logger, "local"):
                logger.setLevel(level)


def create_simple_logger(
    logger_name: str, level: str = LOG_LEVEL, set_level_to_all_loggers: bool = False
) -> logging.Logger:
    """Creates a simple logger with the given name and level. The logger has a single handler that logs to the console.

    Parameters
    ----------
    logger_name : str
        Name of the logger.
    level : str or int
        Level of the logger. Can be a string or an integer. If a string, it should be one of the following: "debug", "info", "warning", "error", "critical". Default level is read from the environment variable LOG_LEVEL.

    Returns
    -------
    logging.Logger
        The logger object.
    """
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]
    logger = logging.getLogger(logger_name)
    logger.local = True
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if set_level_to_all_loggers:
        set_logger_level_to_all_local(level)
    return logger


logger = create_simple_logger(__name__)


def is_jupyter_notebook() -> bool:
    """Checks if the code is being run in a Jupyter notebook.

    Returns
    -------
    bool
        True if the code is being run in a Jupyter notebook, False otherwise.
    """
    is_jupyter = False
    try:
        # noinspection PyUnresolvedReferences
        from IPython import get_ipython

        # noinspection PyUnresolvedReferences
        if get_ipython() is None or "IPKernelApp" not in get_ipython().config:
            pass
        else:
            is_jupyter = True
    except (ImportError, NameError):
        pass
    if is_jupyter:
        logger.debug("Running in Jupyter notebook.")
    else:
        logger.debug("Not running in a Jupyter notebook.")
    return is_jupyter


def create_wandb_logger(
    name: Union[str, None] = None,
    project: Union[str, None] = None,
    config: Union[dict[str, any], None] = None,
    tags: Union[list[str], None] = None,
    notes: str = "",
    group: Union[str, None] = None,
    job_type: str = "",
    logger: Union[logging.Logger, None] = None,
) -> Any:
    """Creates a new run on Weights & Biases and returns the run object.

    Parameters
    ----------
    project : str | None, optional
        The name of the project. If None, it must be provided in the config. Default is None.
    name : str | None, optional
        The name of the run. If None, it must be provided in the config. Default is None.
    config : dict[str, any] | None, optional
        The configuration to be logged. Default is None. If `project` and `name` are not provided, they must be present in the config.
    tags : list[str] | None, optional
        The tags to be added to the run. Default is None.
    notes : str, optional
        The notes to be added to the run. Default is "".
    group : str | None, optional
        The name of the group to which the run belongs. Default is None.
    job_type : str, optional
        The type of job. Default is "train".
    logger : logging.Logger | None, optional
        The logger to be used by the object. If None, a simple logger is created using `create_simple_logger`. Default is None.

    Returns
    -------
    wandb.Run
        The run object.
    """
    import wandb

    logger = logger or create_simple_logger("create_wandb_logger")
    if config is None:
        logger.debug("No config provided. Using an empty config.")
        config = {}

    if name is None and "name" not in config.keys():
        m = "Run name must be provided either as an argument or in the config."
        logger.error(m)
        raise ValueError(m)

    if project is None and "project" not in config.keys():
        m = "Project name must be provided either as an argument or in the config."
        logger.error(m)
        raise ValueError(m)

    # If the arguments are provided, they take precedence over the config
    name = name or config.get("name")
    project = project or config.get("project")
    notes = notes or config.get("notes")
    tags = tags or config.get("tags")
    group = group or config.get("group")
    job_type = job_type or config.get("job_type")

    logger.info(
        f"Initializing Weights & Biases for project {project} with run name {name}."
    )
    wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        group=group,
        job_type=job_type,
    )
    return wandb


def set_publish_plotly_template(mode="light") -> None:
    """Sets the plotly template for publication-ready plots."""
    import plotly.graph_objects as go
    import plotly.io as pio

    text_color = "black" if mode == "light" else "white"
    background_color = "white" if mode == "light" else "black"

    pio.renderers.default = "notebook"
    font_family = "Times New Roman"

    def get_font_dict(size, color=text_color):
        return dict(
            size=size,
            color=color,
            family=font_family,
            weight="bold",
            variant="small-caps",
        )

    pio.templates["publish"] = go.layout.Template(
        layout=go.Layout(
            title=dict(
                font=get_font_dict(24),
            ),
            legend=dict(
                font=get_font_dict(18),
            ),
            xaxis=dict(
                title=dict(
                    font=get_font_dict(18),
                ),
                tickfont=get_font_dict(16),
            ),
            yaxis=dict(
                title=dict(
                    font=get_font_dict(18),
                ),
                tickfont=get_font_dict(16),
            ),
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
        ),
    )
    pio.templates.default = "publish"
    logger.info(f"Plotly template ready for publication {mode} mode.")


def set_publish_matplotlib_template(mode="light") -> None:
    """Sets the matplotlib template for publication-ready plots."""
    import matplotlib.pyplot as plt

    text_color = "black" if mode == "light" else "white"
    background_color = "white" if mode == "light" else "black"
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelcolor": text_color,
            "axes.labelsize": 18,
            "axes.labelweight": "bold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.color": text_color,
            "ytick.color": text_color,
            "axes.grid": True,
            "axes.facecolor": background_color,
            "figure.facecolor": background_color,
            "figure.titlesize": 20,
            "figure.titleweight": "bold",
            "grid.color": "#948b72",
            "grid.linewidth": 1,
            "grid.linestyle": "--",
            "axes.edgecolor": text_color,
            "axes.linewidth": 0.5,
        }
    )
    logger.info(f"Matplotlib template ready for publication {mode} mode.")
