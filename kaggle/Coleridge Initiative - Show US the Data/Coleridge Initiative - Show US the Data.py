import os
import re
import pandas as pd
import numpy as np
import spacy


def extract(tok, exclusion_phrases=[]):
    """Looks backwards and forwards from the token to extract a dataset mention.
    The optional exclusion_phrases argument is a list of strings that, if any one is present,
    no dataset mention is returned.
    """
    matches = []
    special_tokens1 = ['a', 'the', 'in', 'on', 'of', 'across', 'and']
    special_tokens2 = ['-', '\n', '\"', "'s"]
    special_tokens3 = ['-', "'s", '19']
    special_tokens4 = ['center', 'centers', 'centre', 'centres', 'program', 'programmes', \
                       'division', 'branch', 'branches', 'administration']

    match = [tok.text]

    # if the token is not at the beginning of the sentence...
    if not tok.is_sent_start:
        ptr = -1
        prev_tok = tok.nbor(ptr)

        # Add words to the left of the token.
        while not prev_tok.text in special_tokens1 and (prev_tok.text.istitle() or prev_tok.text.isupper()) or prev_tok.text in special_tokens1 + special_tokens3 and not prev_tok.is_sent_start:
            match.insert(0, prev_tok.text)

            if not prev_tok.is_sent_start:
                ptr = ptr - 1
                prev_tok = tok.nbor(ptr)
            else:
                break

        # Chop off extraneous words
        while match[0][0].islower() or match[0].lower() in special_tokens1 or match[0] in special_tokens2:
            match.pop(0)

        # If the token is not at the end of the sentence...
        if not tok.is_sent_end:
            ptr = 1
            nxt_tok = tok.nbor(ptr)

            # Add words to the right of the token
            while nxt_tok.text.istitle() or nxt_tok.text in special_tokens1 + special_tokens3:
                match.append(nxt_tok.text)

                if not nxt_tok.is_sent_end:
                    ptr = ptr + 1
                    nxt_tok = tok.nbor(ptr)
                else:
                    break

            # Chop off extraneous words
            while match[-1][0].islower() or match[-1].lower() in special_tokens1 or match[-1] in special_tokens2 or '-' in match and (match.index(tok.text) < match.index('-')):
                match.pop(-1)

            # If the dataset mention is at least two words long and the last word is not an
            # desirable word...
            if len(match) > 2 and not match[-1].lower() in special_tokens4:
                match_text = ' '.join(match)

                # If no exclusion phrases are present...
                if all([x not in match_text.lower() for x in exclusion_phrases]):
                    # remove extra spaces for hyphens and apostrophe s that were
                    # introduced in the ' ' .join
                    match_text = match_text.replace(' -', '-')
                    match_text = match_text.replace('- ', '-')
                    match_text = match_text.replace(" 's", "'s")

                    # split if two datasets, which sometimes happens
                    if 'and the' in match_text:
                        two_matches = match_text.split(' and the ')

                        if len(two_matches[0].split(' ')) > 2:
                            matches.append(two_matches[0])

                        if len(two_matches[1].split(' ')) > 2:
                            matches.append(two_matches[1])
                    else:
                        matches.append(match_text)

    matches = list(set(matches))

    return matches


def extractor(txt):
    """Executes the extractor functions as the keywords are encountered in the text"""

    def chunks(text, n):
        """Used to chunk text if it is longer than SpaCy's allowed nlp.max_length"""
        for idx in range(0, len(text), n):
            yield text[idx:idx + n]

    all_matches = []

    # put a space on either side of a hyphen so that it will be tokenized separately
    txt = re.sub(r'(.)-(.)', r'\1 - \2', txt)

    # put a space on either side of a year that is touching other text so that the year
    # will be tokenized separately
    txt = re.sub(r'(.)((19|20)[0-9]{2})(.)', r'\1 \2 \4', txt)

    for txt_chunk in chunks(txt, nlp.max_length):
        for doc in nlp.pipe([txt_chunk]):
            for tok in doc:
                if tok.text in keywords.keys():
                    all_matches.extend(extract(tok, keywords[tok.text]))

    all_matches = list(set(all_matches))

    return all_matches


def clean_text(txt):
    """Clean the text as specified in the competition instructions"""
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


def get_string_matches(txt):
    """Find matches of strings in hi_freq_datasets"""
    matches = []

    for label in hi_freq_datasets:
        # X out all matches of this label so that shorter strings representing the same
        # dataset do not hit the mention that has already been accounted for
        txt, n_subs = re.subn(label, 'X' * len(label), txt)

        if n_subs > 0:
            matches.append(clean_text(label))

    matches = list(set(matches))

    return matches


keywords = {'Study': ['of study', 'case study'], 'Survey': ['geologic', 'of survey', 'system'], \
            'Database': [], 'Dataset': [], 'Archive': [], 'Assessment': [], 'Catalog': [], \
            'Collection': [], 'Registry': [], 'Initiative': []}
nlp = spacy.load('en_core_web_sm')

nlp.add_pipe(nlp.create_pipe('sentencizer'))
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')
nlp.remove_pipe('ner')

nlp.pipeline


file_dir = '../input/coleridgeinitiative-show-us-the-data/test'

file_id = []
text = []

files = os.listdir(file_dir)

for file in files:
    file_id.append(os.path.splitext(os.path.basename(file))[0])
    tmp_text = ' '.join(pd.read_json(os.path.join(file_dir, file), orient='records')['text'])
    text.append(tmp_text)

df = pd.DataFrame(data={'Id':file_id, 'raw_text':text})
preds_list = df['raw_text'].apply(extractor)
hi_freq_datasets = []

for preds in preds_list:
    hi_freq_datasets.extend(preds)

df_temp = pd.Series(hi_freq_datasets).value_counts().to_frame('counts')
df_temp = df_temp[df_temp.index.str.len() > 0].copy() # remove blanks
thresh = np.percentile(df_temp, 95)
hi_freq_datasets = list(df_temp[df_temp.counts >= thresh].index)
df_train = pd.read_csv('../input/coleridgeinitiative-show-us-the-data/train.csv')

known_datasets = pd.unique(df_train[['dataset_title', 'dataset_label']].values.ravel('K')).tolist()
known_datasets = known_datasets + ['National Postsecondary Student Aid Study', \
    'Schools and Staffing Survey', 'National Survey of College Graduates', \
    'Framingham Heart Study', 'National Survey of Recent College Graduates', \
    'Program for International Student Assessment', 'Health and Retirement Study', \
    'Private School Universe Survey', 'Teaching and Learning International Survey', \
    'National Crime Victimization Survey', 'International Mathematics and Science Study', \
    'Consumer Expenditure Survey', 'Current Population Survey', 'American Community Survey', \
    'National Health Interview Survey', 'Progress in International Reading Literacy Study', \
    'Scientists and Engineers Statistical Data System', \
    'International Best Track Archive for Climate Stewardship IBTrACS', \
    'Sea Lake and Overland Surges from Hurricanes SLOSH', \
    'National Longitudinal Survey of Youth', 'National Study of Postsecondary Faculty', \
    'National Educational Longitudinal Study', \
    'Global Initiative on Sharing All Influenza Data', \
    'National Longitudinal Study of Adolescent Health', \
    'National Longitudinal Study of Adolescent to Adult Health', 'ADCIRC', \
    'Private School Survey', 'National Land Cover', 'COVID-19 Open Data', \
    'National Education Longitudinal Study of 1988', \
    'High School Longitudinal Study of 2009', \
    'National Health and Nutrition Examination Survey']

hi_freq_datasets = hi_freq_datasets + known_datasets
hi_freq_datasets = list(set(hi_freq_datasets))
hi_freq_datasets.sort(key=len, reverse=True)
df['PredictionString'] = df['raw_text'].apply(get_string_matches).str.join('|')
df[['Id', 'PredictionString']].to_csv('submission.csv', index=False)