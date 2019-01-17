from CRF_NER import CRF_NER
import pandas as pd

# Create Gazateer
gazateer = pd.read_csv('gazateer/gazateer.csv')
gazateer = dict(zip(gazateer['entities'].tolist(), gazateer['categories'].tolist()))

# Get documents
documents = [str(x) for x in pd.read_csv('data/techCorpus.csv')['text'].tolist()]

# Training Model
ner_crf = CRF_NER(gazateer)
ner_crf.train(documents)

# Predictions
ner_crf.predict('Peter is looking to work for Google in Ireland where the Government is banning iphones')
