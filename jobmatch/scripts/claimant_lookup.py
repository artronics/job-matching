import json

import pandas as pd
from tqdm import tqdm

from jobmatch import DATA_DIR
from jobmatch.scripts.claimant import match_text_to_soc
from jobmatch.scripts.soc_extraction import make_soc_job_title_with_id


def make_unified_unique_job_titles():
    claimant_csv = DATA_DIR / "claimant.csv"
    col1 = 'jobTitle'
    col2 = 'preferredjobs'
    df = pd.read_csv(claimant_csv, usecols=[col1, col2])

    all_values = []

    for col in [col1, col2]:
        df[col] = df[col].fillna('')  # Handle NaN just in case
        for cell in df[col]:
            items = [item.strip().lower() for item in cell.split(',') if item.strip()]
            all_values.extend(items)

    return list(set(all_values))


def make_soc_title_to_id_lookup():
    socs = make_soc_job_title_with_id()
    lookup = {}
    for item in socs:
        lookup.update({item['title']: item['id']})

    return lookup

def add_soc_codes(job_titles: [str]) -> dict:
    matches_limit = 5
    lookup = {}
    soc_to_id = make_soc_title_to_id_lookup()
    for title in tqdm(job_titles, desc="Matching SOC codes"):
        titles, _ = match_text_to_soc(title, matches_limit)
        try:
            socs = [{'id': soc_to_id[t], 'title': t} for t in titles]
        # The original file from gov is encoded in cp1252. Some titles have bad utf-8 characters and causes errors
        # We ignore them
        except KeyError:
            continue
        lookup.update({title: socs })

    return lookup


def main():
    unique_titles = make_unified_unique_job_titles()
    lookup = add_soc_codes(unique_titles)
    with open('claimant_lookup.json', 'w', encoding='utf-8') as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)



if __name__ == '__main__':
    main()
