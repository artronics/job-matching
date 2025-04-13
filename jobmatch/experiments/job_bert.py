import unittest

import numpy as np
from sentence_transformers.util import cos_sim

import jobmatch.advert as advert
import jobmatch.skills as skills
import jobmatch.soc as soc
from jobmatch.cache import Cache
from jobmatch.model_jobbert import JobBert
from report import *

cache = Cache()


class TestJobBert(unittest.TestCase):
    def setUp(self):
        self.adverts = advert.load_dataset()
        self.job_bert = JobBert()
        self.soc_titles = np.array([t.get_title() for t in soc.load_soc_codes().get_job_titles()])
        self.skills = skills.make_onet_skills().dataset

        embeddings_name = "test_soc_codes_titles_embeddings"
        if cache.exists(embeddings_name):
            self.title_embeddings = cache.load_np(embeddings_name)
        else:
            self.title_embeddings = self.job_bert.encode(self.soc_titles)
            cache.store_np(embeddings_name, self.title_embeddings)

        embeddings_name = "test_onet_skills_embeddings"
        if cache.exists(embeddings_name):
            self.skills_embeddings = cache.load_np(embeddings_name)
        else:
            self.skills_embeddings = self.job_bert.encode(self.skills)
            cache.store_np(embeddings_name, self.skills_embeddings)

    def match_skills(self, advert_text: str, skills_embeddings):
        advert_embedding = self.job_bert.encode([advert_text])[0]
        similarities = cos_sim(advert_embedding, skills_embeddings)[0].numpy()
        sorted_matches = sorted(zip(self.skills, similarities), key=lambda x: x[1], reverse=True)

        return sorted_matches

    def match(self, advert_text: str, titles_embeddings):
        advert_embedding = self.job_bert.encode([advert_text])[0]
        similarities = cos_sim(advert_embedding, titles_embeddings)[0].numpy()
        sorted_matches = sorted(zip(self.soc_titles, similarities), key=lambda x: x[1], reverse=True)

        return sorted_matches

    def test_soc_match(self):
        random_advert: Advert = self.adverts.choice()

        advert_text = f"{random_advert.title}:\n{random_advert.content}"
        matches = self.match(advert_text, self.title_embeddings)

        report = Report("JobBERT match", [self.make_report_item(random_advert, matches)])
        print(generate_markdown_report(report))

    def test_match_many(self):
        count = 20
        report_items: [ReportItem] = []
        for i in range(count):
            random_advert: Advert = self.adverts.choice()

            advert_text = f"{random_advert.title}:\n{random_advert.content}"
            matches = self.match(advert_text, self.title_embeddings)

            report_items.append(self.make_report_item(random_advert, matches))

        report = Report(title=f"JobBERT SOC codes {20} matches", items=report_items)
        write_markdown_report(f"jobbert_soc_titles_matches.md", report)

    def test_just_job_titles(self):
        count = 20
        report_items: [ReportItem] = []
        for i in range(count):
            random_advert: Advert = self.adverts.choice()
            matches = self.match(random_advert.title.get_title(), self.title_embeddings)

            report_items.append(ReportItem(
                title=random_advert.title.get_title(),
                results=[ReportResult(title="Top 5 job title matches for this title",
                                      content=make_similarities_table(matches))]
            ))

        report = Report(
            title=f"JobBERT SOC codes {20} matches",
            overview="""In this experiment we only try to match the job title of an advert.
            This can be used in claimant data. In our claimant data we have a list of `preferredJobs` with no description.
            The experiment demonstrate what happens when we remove the context around a job title i.e. job description.
            """,
            items=report_items)
        write_markdown_report(f"jobbert_soc_titles_with_only_job_titles.md", report)

    def test_jobbert_for_skills_many(self):
        count = 20
        report_items: [ReportItem] = []
        for i in range(count):
            random_advert: Advert = self.adverts.choice()

            advert_text = f"{random_advert.title}:\n{random_advert.content}"
            matches = self.match_skills(advert_text, self.skills_embeddings)

            report_items.append(self.make_report_item(random_advert, matches))

        report = Report(
            title=f"JobBERT ONET skills {20} matches",
            overview="""In this experiment we use JobBERT to find similarities between job advert full text
            and ONET skills database.""",
            items=report_items)
        write_markdown_report(f"jobbert_onet_skills_matches.md", report)

    @staticmethod
    def make_report_item(ad: Advert, matches) -> ReportItem:
        table = make_similarities_table(matches)

        return ReportItem(
            title=ad.title.get_title(),
            content=ad.content,
            results=[ReportResult(title="Top 5 job title matches for the advert", content=table)]
        )
