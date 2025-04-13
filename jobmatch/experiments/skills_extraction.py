import unittest

from sentence_transformers.util import cos_sim

from jobmatch import job_datasets as datasets
from jobmatch.embeddings import make_embeddings
from jobmatch.experiments.report import Report, write_markdown_report, ReportItem, make_similarities_table, ReportResult
from jobmatch.llm import ContextSkillExtraction
from jobmatch.models import Advert

_model = ContextSkillExtraction()

title = "Extracting skills based on ONET dataset"
overview = f"""
In this experiment, we use the `{_model.name}` model to sort ONET sills dataset based on a given job advert""".strip()

usage = """
* Tagging job adverts
* Classifying job adverts, and finding relevance to SOC codes
""".strip()


class TestJobBertSkillsExtraction(unittest.TestCase):
    cache_prefix = "context_skills_onet_"

    def setUp(self):
        self.model = _model

        self.adverts = datasets.make_job_adverts_dataset()
        # soc_ds = datasets.make_soc_job_titles().dataset
        # self.soc_titles = np.array([t.get_job_title().title for t in soc_ds])
        self.onet_skills = datasets.make_onet_skills_dataset().dataset

    def test_match_job_title_to_soc(self):
        count = 20
        report_items: [ReportItem] = []
        skills_emb = make_embeddings(self.model, f"onet_only_skills", self.onet_skills)

        for i in range(count):
            random_advert: Advert = self.adverts.choice()
            advert_text = str(random_advert)

            ad_emb = self.model.encode([advert_text])[0]

            similarities = cos_sim(ad_emb, skills_emb)[0].numpy()
            sorted_matches = sorted(zip(self.onet_skills, similarities), key=lambda x: x[1], reverse=True)

            report_items.append(self.make_report_item(random_advert, sorted_matches))

        report = Report(title, overview=overview, usage=usage, items=report_items)
        write_markdown_report(f"{title.replace(' ', '_')}.md", report)

    @staticmethod
    def make_report_item(ad: Advert, matches) -> ReportItem:
        table = make_similarities_table(matches, limit=10)

        return ReportItem(
            title=str(ad.title),
            content=str(ad),
            results=[ReportResult(title="Top 10 skills that matches this job advert", content=table)]
        )
