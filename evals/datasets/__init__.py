"""The datasets a trial can run on."""

from evals.dataset import Dataset
from evals.datasets.company import COMPANY
from evals.datasets.company_synonyms import COMPANY_SYNONYMS
from evals.datasets.movies import MOVIES
from evals.datasets.skills import SKILLS

DATASETS: dict[str, Dataset] = {
    "movies": MOVIES,
    "company": COMPANY,
    "company_synonyms": COMPANY_SYNONYMS,
    "skills": SKILLS,
}
