import copy
from dataclasses import dataclass
from typing import List


@dataclass
class JobTitle:
    title: str

    def __str__(self):
        return self.title


@dataclass
class JobTitleClassificationPath:
    path: List[str]
    "The path from the root of taxonomy hierarchy to the job title"

    def get_job_title(self) -> JobTitle:
        return JobTitle(self.path[-1])

    def __str__(self):
        return ' > '.join(self.path)


@dataclass
class Advert:
    title: JobTitle
    contents: List[str]

    def __str__(self):
        return "\n\n".join([str(self.title)] + self.contents)


class Matching:
    """Response object contains **sorted** titles. Scores are provided as a separate list."""
    titles: [str]
    scores: [float]

    def __init__(self, titles: [str], scores: [float]):
        self.titles = titles
        self.scores = scores

    def as_dict(self):
        return {'titles': self.titles, 'scores': self.scores}


@dataclass
class TitleMatchingResponse:
    title: str
    matches: Matching

    def __init__(self, title: str, matches: Matching):
        self.title = title
        self.matches = matches

    def as_dict(self):
        return {
            self.title : self.matches.as_dict(),
        }


@dataclass
class PreferredJobs:
    yesNo: bool
    values: List[str]


@dataclass
class Qualifications:
    values: List[str]


@dataclass
class ContentData:
    hasWorkHistory: bool
    modesOfTravel: List[str]
    preferredJobs: PreferredJobs
    confidenceLevel: str
    hasCV: bool
    canDoJobs: List[str]
    qualifications: Qualifications
    socs: dict = None


@dataclass
class Claimant:
    contentData: ContentData

    @staticmethod
    def from_defaults() -> 'Claimant':
        return Claimant(
            contentData=ContentData(
                hasWorkHistory=False,
                modesOfTravel=["WALK"],
                preferredJobs=PreferredJobs(
                    yesNo=False,
                    values=[]
                ),
                confidenceLevel="NOT_CONFIDENT",
                hasCV=False,
                canDoJobs=[],
                qualifications=Qualifications(
                    values=[]
                )
            )
        )

    def update_socs(self, socs: dict) -> 'Claimant':
        cloned = copy.deepcopy(self)
        cloned.socs = socs
        return cloned
