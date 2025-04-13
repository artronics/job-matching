import unittest

import jobmatch.skills as skills
import jobmatch.soc as soc
from jobmatch.model_jobbert import JobBert
from jobmatch.model_skill_extraction import JobBertSkillExtraction
from sentence_transformers.util import cos_sim

import jobmatch.advert as advert
from jobmatch.advert import Advert
from jobmatch.cache import Cache
from report import *

cache = Cache()


class TestJobBertSkillsExtraction(unittest.TestCase):
    def setUp(self):
        self.model = JobBertSkillExtraction()
        self.adverts = advert.load_dataset()
        self.skills = skills.make_onet_skills().dataset

        embeddings_name = "test_onet_skills_embeddings_extraction_model"
        if cache.exists(embeddings_name):
            self.skills_embeddings = cache.load_np(embeddings_name)
        else:
            self.skills_embeddings = self.model.encode(self.skills)
            cache.store_np(embeddings_name, self.skills_embeddings)

    def match_skills(self, advert_text: str, skills_embeddings):
        advert_embedding = self.model.encode([advert_text])[0]
        similarities = cos_sim(advert_embedding, skills_embeddings)[0].numpy()
        sorted_matches = sorted(zip(self.skills, similarities), key=lambda x: x[1], reverse=True)

        return sorted_matches

    def test_jobbert_for_skills_many(self):
        count = 20
        report_items: [ReportItem] = []
        for i in range(count):
            random_advert: Advert = self.adverts.choice()

            advert_text = f"{random_advert.title}:\n{random_advert.contents}"
            matches = self.match_skills(advert_text, self.skills_embeddings)

            report_items.append(self.make_report_item(random_advert, matches))

        report = Report(
            title=f"JobBERT (skills extraction) ONET skills {20} matches",
            overview="""In this experiment we use JobBERT skills extraction to find similarities between job advert full text
            and ONET skills database.""",
            items=report_items)
        write_markdown_report(f"jobbert_onet_skills_matches_with_skills_extraction.md", report)

    @staticmethod
    def make_report_item(ad: Advert, matches) -> ReportItem:
        table = make_similarities_table(matches)

        return ReportItem(
            title=ad.title.get_title(),
            content=ad.contents,
            results=[ReportResult(title="Top 5 skills matches for the advert", content=table)]
        )
