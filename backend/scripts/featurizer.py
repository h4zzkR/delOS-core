from core.nlu.featurizers.transformer_featurizer import SentenceFeaturizer

if __name__ == "__main__":
    text = 'turn on the light please'
    f = SentenceFeaturizer()
    print(f.featurize(text))