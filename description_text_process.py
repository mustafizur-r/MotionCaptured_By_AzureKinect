import pandas as pd
import spacy
from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin

# Load Spacy model
nlp = spacy.load('en_core_web_sm')


def process_text(sentence):
    """Processes a single sentence, tokenizing and POS tagging."""
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []

    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)

    return word_list, pos_list


def process_humanml3d(corpus):
    """Processes a dataset where each row contains multiple sentences in the 'caption' column."""
    text_save_path = './dataset-making-process/pose_data_raw/texts'

    for i in tqdm(range(len(corpus))):
        captions = str(corpus.iloc[i]['caption']).split('\n')  # Convert to str and split sentences
        start = corpus.iloc[i].get('from', 0.0)  # Use default value if missing
        end = corpus.iloc[i].get('to', 0.0)  # Use default value if missing
        name = str(corpus.iloc[i].get('new_joint_name', 'unknown')).strip()  # Convert to str, use 'unknown' if missing

        if name == 'nan' or name == '':
            name = 'unknown'  # Fallback for missing names

        file_path = pjoin(text_save_path, name.replace('npy', 'txt'))  # Ensure name is a string

        with cs.open(file_path, 'a+', encoding='utf-8') as f:
            for caption in captions:
                caption = caption.strip()
                if caption:
                    word_list, pos_list = process_text(caption)
                    tokens = ' '.join([f'{word_list[i]}/{pos_list[i]}' for i in range(len(word_list))])
                    f.write(f'{caption}#{tokens}#{start}#{end}\n')


def process_kitml(corpus):
    """Processes a dataset where each row contains multiple sentences in the 'desc' column."""
    text_save_path = './dataset-making-process/kit_mocap_dataset/texts'

    for i in tqdm(range(len(corpus))):
        captions = str(corpus.iloc[i]['desc']).split('\n')  # Convert to str and split sentences
        start = 0.0
        end = 0.0
        name = str(corpus.iloc[i].get('data_id', 'unknown')).strip()  # Convert to str, use 'unknown' if missing

        if name == 'nan' or name == '':
            name = 'unknown'

        file_path = pjoin(text_save_path, name + '.txt')  # Ensure name is a string

        with cs.open(file_path, 'a+', encoding='utf-8') as f:
            for caption in captions:
                caption = caption.strip()
                if caption:
                    word_list, pos_list = process_text(caption)
                    tokens = ' '.join([f'{word_list[i]}/{pos_list[i]}' for i in range(len(word_list))])
                    f.write(f'{caption}#{tokens}#{start}#{end}\n')


if __name__ == "__main__":
    corpus = pd.read_csv('./dataset-making-process/pose_data_raw/desc_final.csv')
    process_humanml3d(corpus)
