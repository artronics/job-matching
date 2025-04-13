import unittest

import numpy as np
from sentence_transformers.util import cos_sim

import jobmatch.job_datasets as datasets
from jobmatch.embeddings import make_embeddings
from jobmatch.experiments.report import ReportItem, Report, write_markdown_report, make_similarities_table, ReportResult
from jobmatch.llm import JobBertV2
from jobmatch.models import Advert

_model = JobBertV2()

title = "Finding the best SOC match for a given job posting"
overview = f"""
In this experiment, we use the `{_model.name}` model to identify the Standard Occupational Classification (SOC) code that best 
matches a given job advertisement. Each experiment involves selecting a random job advert from the 
`udacity/armenian_online_job_postings` dataset. The input to the model contains a text composed of job title and the 
job description. The model then returns the top five SOC codes that are most similar to the input, ranked 
by relevance.""".strip()

usage = """
This model works really well as it is. We can use it for:
* Choosing the best SOC code, when migrating existing database
* Give user suggestions for the job title
* Tagging jobs for the semantic search capability
""".strip()


class TestJobBertV2AdvertTitleToSoc(unittest.TestCase):
    cache_prefix = "ad_soc_"

    def setUp(self):
        self.model = _model

        self.adverts = datasets.make_job_adverts_dataset()
        soc_ds = datasets.make_soc_job_titles().dataset
        self.soc_titles = np.array([t.get_job_title().title for t in soc_ds])

    def test_match_job_advert_title_to_soc(self):
        count = 20
        report_items: [ReportItem] = []
        soc_emb = make_embeddings(self.model, f"soc_only_titles", self.soc_titles)

        for i in range(count):
            random_advert: Advert = self.adverts.choice()
            advert_text = str(random_advert)

            ad_emb = self.model.encode([advert_text])[0]

            similarities = cos_sim(ad_emb, soc_emb)[0].numpy()
            sorted_matches = sorted(zip(self.soc_titles, similarities), key=lambda x: x[1], reverse=True)

            report_items.append(self.make_report_item(random_advert, sorted_matches))

        report = Report(title, overview=overview, usage=usage, items=report_items)
        write_markdown_report(f"{title.replace(' ', '_')}.md", report)

    @staticmethod
    def make_report_item(ad: Advert, matches) -> ReportItem:
        table = make_similarities_table(matches)

        return ReportItem(
            title=str(ad.title),
            content=str(ad),
            results=[ReportResult(title="Top 5 SOC codes matching the given job title", content=table)]
        )
