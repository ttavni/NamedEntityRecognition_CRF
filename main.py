from CRF_NER import CRF_NER
import pandas as pd

if __name__ == "__main__":

	# Create Gazateer
	gazateer = pd.read_csv('gazateer/gazateer.csv')
	gazateer = dict(zip(gazateer['categories'].tolist(), gazateer['entities'].tolist()))

	# Get documents
	documents = [str(x) for x in pd.read_csv('data/techCorpus.csv')['text'].tolist()]

	# Training Model
	ner_crf = CRF_NER(gazateer)
	ner_crf.train(documents)

	# Predictions
	ner_crf.predict('')
