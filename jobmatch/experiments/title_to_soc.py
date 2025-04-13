import unittest

import numpy as np
from sentence_transformers.util import cos_sim

import jobmatch.job_datasets as datasets
from jobmatch.embeddings import make_embeddings
from jobmatch.experiments.report import ReportItem, Report, write_markdown_report, make_similarities_table, ReportResult
from jobmatch.llm import JobBertV2
from jobmatch.models import Advert, JobTitle

_model = JobBertV2()

title = "Finding the best SOC match for a given title"
overview = f"""
In this experiment, we use the `{_model.name}` model to identify the Standard Occupational Classification (SOC) code that best 
matches a given title. Each experiment involves selecting a random job advert from the 
`udacity/armenian_online_job_postings` dataset. The input to the model **only** contains the job. So the main difference,
 is that with this experiment the input is only a job title and no more context is provided. The model then 
returns the top five SOC codes that are most similar to the input, ranked by relevance.""".strip()

usage = """
This model works really well as it is. We can use it for:
* Choosing the best SOC code, when migrating claimant data
* Give user suggestions for the job title (claimant input)
* Tagging claimants or job seekers for the semantic search capability
""".strip()


class TestJobBertV2TitleToSoc(unittest.TestCase):
    cache_prefix = "title_soc_"

    def setUp(self):
        self.model = _model

        self.adverts = datasets.make_job_adverts_dataset()
        soc_ds = datasets.make_soc_job_titles().dataset
        self.soc_titles = np.array([t.get_job_title().title for t in soc_ds])

    def test_match_job_title_to_soc(self):
        count = 20
        report_items: [ReportItem] = []
        soc_emb = make_embeddings(self.model, f"soc_only_titles", self.soc_titles)

        for i in range(count):
            random_advert: Advert = self.adverts.choice()
            title_text = random_advert.title.title

            ad_emb = self.model.encode([title_text])[0]

            similarities = cos_sim(ad_emb, soc_emb)[0].numpy()
            sorted_matches = sorted(zip(self.soc_titles, similarities), key=lambda x: x[1], reverse=True)

            report_items.append(self.make_report_item(random_advert, sorted_matches))

        report = Report(title, overview=overview, usage=usage, items=report_items)
        write_markdown_report(f"{title.replace(' ', '_')}.md", report)

    def test_match_job_title_to_soc_for_claimant(self):
        report_items: [ReportItem] = []
        soc_emb = make_embeddings(self.model, f"soc_only_titles", self.soc_titles)

        job_titles = [
            "Warehouse",
            "Data Warehouse Engineer"
            "Lift truck worker in warehouse",
            # Source Reddit: Most ridiculous job titles you've seen :)
            "Spirit guidance counsellor",
            "Head of Boning",
            "Surface Engineer",
            "Planetary Protection Officer",
            "Chief inspiration officer",
            "Director of First Impressions"]
        for job_title in job_titles:
            ad_emb = self.model.encode([job_title])[0]

            similarities = cos_sim(ad_emb, soc_emb)[0].numpy()
            sorted_matches = sorted(zip(self.soc_titles, similarities), key=lambda x: x[1], reverse=True)

            report_item = self.make_report_item(Advert(title=JobTitle(job_title), contents=[]), sorted_matches)
            report_items.append(report_item)

        report_title = "Finding the best SOC match for claimant input"
        report = Report(report_title, overview=overview, usage=usage, items=report_items)
        write_markdown_report(f"{report_title.replace(' ', '_')}.md", report)

    @staticmethod
    def make_report_item(ad: Advert, matches) -> ReportItem:
        table = make_similarities_table(matches)

        return ReportItem(
            title=str(ad.title),
            content="",
            results=[ReportResult(title="Top 5 SOC codes matching the given job title", content=table)]
        )
