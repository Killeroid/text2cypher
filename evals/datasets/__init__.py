"""The datasets a trial can run on."""

from evals.dataset import Dataset
from evals.datasets.company import COMPANY
from evals.datasets.movies import MOVIES

DATASETS: dict[str, Dataset] = {"movies": MOVIES, "company": COMPANY}
