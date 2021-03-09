'''
Functionality for loading the trained models, processing texts, and visualizing the output.
Set the working directory to ner_wikiner project root.
'''
from random import shuffle, seed

import spacy
from pathlib import Path
from spacy.tokens import Doc, DocBin
from spacy import displacy

def model_load(path='training/model-best'):
    nlp = spacy.load(path)
    return nlp

def print_docs_sample(docs, sample_size, rseed=981):
    seed(rseed)
    shuffle(docs); docs = docs[:sample_size]
    for doc in docs:
        print(doc)
        print(doc.ents)

def corpus_load(path='corpus/train.spacy', as_text=False):
    nlp = spacy.blank('en')
    doc_bin = DocBin().from_disk(path)
    if as_text: docs = list(str(doc) for doc in doc_bin.get_docs(nlp.vocab))
    else: docs = list(doc_bin.get_docs(nlp.vocab))
    return docs

def visualize_data(model_path='training/model-best', corpus_path='corpus/train.spacy',
                   sample_size = 10, rseed=192):
    nlp = model_load(model_path)
    docs = corpus_load(corpus_path, as_text=True)
    seed(rseed); shuffle(docs)
    docs = docs[:sample_size]
    for i, txt in enumerate(docs):
        doc = nlp(txt)
        html = displacy.render(doc, style='ent', page=True, minify=True)
        fname = f'doc{i}.html'
        Path('visualizations/'+fname).open('w', encoding='utf-8').write(html)

if __name__ == '__main__':
    visualize_data(sample_size=5)
