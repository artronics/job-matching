import json

import pandas as pd

from jobmatch import DATA_DIR


def make_soc_job_title_with_id():
    soc_f = DATA_DIR / "SOC_volume2_utf8.csv"
    df = pd.read_csv(soc_f, usecols=['UNIQUE ID', 'INDEXOCC_-_natural_word_order'], encoding='utf-8')
    df = df.rename(columns={
        'UNIQUE ID': 'id',
        'INDEXOCC_-_natural_word_order': 'title'
    })
    # Remove job title with less than three characters
    df = df[df['title'].str.len() >= 3]
    return df.to_dict(orient='records')



def main():
    socs = make_soc_job_title_with_id()
    with open('soc_title_id.json', 'w', encoding='utf-8') as f:
        json.dump(socs, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
