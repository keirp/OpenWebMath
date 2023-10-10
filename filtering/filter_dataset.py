# coding=utf-8
import argparse
from datasets import load_dataset, Features
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import re

BAD_URLS = [
    'worldwidescience',
    'science.gov',
    'archive.org',
    'scribd.com',
    'unz.com',
    '/profile/',
    '/researcher',
    'noobstarter.com',
    'philpapers.org',
    'thesa.com',
    'beyondhighbrow.com',
    'careyoukeep.com',
    'eevblog.com',
    'happyslide.net',
    'issuu.com',
    'zh-cn.unz.com',
    'vixra.org',
    'medcraveonline.com',
    'sciendo.com',
    'open.library.ubc.ca',
    'eurotrib.com',
    'postthreads.org',
    'jim.bmj.com',
    'wanweibaike.com',
    'hzdr.de',
    '/joursearch/',
    'docplayer.net',
    'bookofmormonblog.org',
    'bradford-delong.com',
    'profiles.stanford.edu',
    'vo.astronet.ru',
    'homainstallationen.at',
    '/author/',
    '/authors/'
    '/serials/'
    'read.dukeupress.edu',
    'thewikipost.org',
    'is.tuebingen.mpg.de',
    'discourse.darkjedibrotherhood.com',
    'springermedizin.de',
    'materials-chain.com',
    'www.unzmag.net',
    'is.mpg.de',
    'hobby8.5ch.net',
    'forums.penny-arcade.com',
    'wowwiki.com',
    '8chan.moe',
    'plosone.org',
    'www.is.mpg.de',
    'feeds.churchturing.org',
    'learn.gcs.edu',
    'mobinuke.com',
    'judithcurry.com',
    'tek-tips.com',
    'skepticforum.com',
    'all_publications',
    '.de/publications',
    'nih.gov',
    'lastfm.it',
    '/commit',
    'vitaminstore',
    'studylib.net',
    'dokumen.pub',
    'manualzz.com',
    'fraser.stlouisfed.org'
]

libretext_good = [
    'math',
    'phys',
    'stats',
]

accented_chars = ['ü', 'ï', 'ö', 'ê', 'ä', 'â', 'ê', 'î', 'û', 'ô', 'è', 'é', 'à']
accented_chars = set(accented_chars)
def has_accented_char(text):
    num_accent = sum([c in accented_chars for c in text.lower()])
    if len(text) == 0:
        return False
    return num_accent / len(text) > 0.015

def count_latex_formulas(text):
    # Remove unwanted patterns
    cleaned_text = re.sub(r'\$\$\\PageIndex\{[^\}]*\}\$\$', '', text)
    cleaned_text = re.sub(r'\$\\PageIndex\{[^\}]*\}\$', '', cleaned_text)
    
    # Pattern for inline and display math
    pattern = r'\$\$[^\$]*\$\$|\$[^\$]*\$'
    
    matches = re.findall(pattern, cleaned_text)
    
    return len(matches)

def filter_data(data):
    metadata = json.loads(data['metadata'])
    perplexity = metadata['extraction_info']['perplexity']
    math_score = metadata['extraction_info']['math_score']
    if perplexity > 15_000:
        return False
    if math_score < 0.17:
        return False
    if 'arxiv-vanity' in data['url']:
        return False
    # Check if /search is in the path
    if '/search' in data['url'] and '//search' not in data['url']:
        return False
    if 'proceedings' in data['url']:
        return False
    if 'bibbase' in data['url']:
        return False
    if 'nrsworld.com' in data['url']:
        return False
    if 'bibtex' in data['url']:
        return False
    if 'issn' in data['url']:
        return False
    if 'arxiv-export' in data['url']:
        return False
    if 'bmjopen' in data['url']:
        return False
    if 'stackexchange.com/users' in data['url']:
        return False
    if 'mathoverflow.net/users' in data['url']:
        return False
    return True

def process_data(datas):
    # Convert datas (which is keys with lists) to a list of dicts
    datas = [dict(zip(datas, t)) for t in zip(*datas.values())]
    new_datas = []
    for data in datas:
        url = data['url']
        text = data['text']

        should_filter = not filter_data(data)

        # Filter out bad urls
        for bad_url in BAD_URLS:
            if bad_url in url:
                should_filter = True
                break

        # Remove any line that has more than one "newcommand"
        lines = text.split('\n')
        new_lines = []
        for line in lines:
            if line.count('newcommand') > 1:
                continue
            new_lines.append(line)
        text = '\n'.join(new_lines)
        data['text'] = text

        # Filter less than 100 characters
        if len(text) < 100:
            should_filter = True

        if 'libretexts' in url:
            # Check if the url is part of the whitelist
            is_whitelist = False
            for good in libretext_good:
                if good in url:
                    is_whitelist = True
                    break
            if not is_whitelist:
                # Throw out if 0 math formulas
                if count_latex_formulas(text) == 0:
                    should_filter = True

        # Filter out accents
        if has_accented_char(text):
            should_filter = True

        if not should_filter:
            new_datas.append(data)

    # Transform back to a dict of lists
    new_datas = {k: [d[k] for d in new_datas] for k in new_datas[0]}
    return new_datas

def main(args):
    dataset = load_dataset(args.input, split='train')
    print(dataset)

    # Filter the dataset
    filtered_dataset = dataset.map(process_data, num_proc=args.n_processes, batched=True)
    print(filtered_dataset)

    # Save the dataset
    filtered_dataset.save_to_disk(args.output_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Huggingface dataset name")
    parser.add_argument("--output_dataset", type=str, required=True, help="path to save dataset")
    parser.add_argument("--n_processes", type=int, default=32, help="Number of processes to use")

    args = parser.parse_args()
    main(args)