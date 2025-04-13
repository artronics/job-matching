from dataclasses import dataclass


@dataclass
class JobTitle:
    title: str


@dataclass
class JobTitleClassificationPath:
    path: [str]
    "The path from the root of taxonomy hierarchy to the job title"

    def get_job_title(self) -> JobTitle:
        return JobTitle(self.path[-1])

    def __str__(self):
        return ' > '.join(self.path)


@dataclass
class Advert:
    title: JobTitle
    contents: [str]
