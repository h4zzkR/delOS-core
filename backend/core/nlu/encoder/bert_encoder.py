import json
import os
from typing import List, Iterable, Union

import numpy as np
import requests
import torch
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import nn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = ROOT_DIR[:ROOT_DIR.find('backend')]

with open(os.path.join(ROOT_DIR, 'backend/configuration/config.json')) as f:
    NLU_ENCODER_CONFIG = json.load(f)
    NLU_ENCODER_CONFIG = NLU_ENCODER_CONFIG['encoder_core_module']


class SentenceTransformerExtended(SentenceTransformer):
    """
    Расширение класса SentenceTransformer для изменения
    возвращаемых значений
    """

    def __init__(self, model_name_or_path: str = None, modules: Iterable[nn.Module] = None, device: str = 'cuda'):
        super().__init__(model_name_or_path, modules, device)

    def decode(self, text):
        return self._first_module().tokenizer.decode(text)

    def featurize(self, sentences: Union[str, List[str], List[int]], batch_size: int = 8, \
                  convert_to_numpy: bool = True, convert_to_tensor: bool = False, \
                  is_pretokenized: bool = False) -> List[ndarray]:
        self.eval()
        if isinstance(sentences, str):  ##Individual sentence
            sentences = [sentences]

        all_embeddings = []
        all_embed_sequences = []
        if is_pretokenized:
            sentences_tokenized = sentences
        else:
            sentences_tokenized = [self.tokenize(sen) for sen in sentences]

        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch_tokens = []

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, len(sentences))

            longest_seq = 0

            for idx in length_sorted_idx[batch_start: batch_end]:
                tokens = sentences_tokenized[idx]
                longest_seq = max(longest_seq, len(tokens))
                batch_tokens.append(tokens)

            features = {}
            for text in batch_tokens:
                sentence_features = self.get_sentence_features(text, longest_seq)

                for feature_name in sentence_features:
                    if feature_name not in features:
                        features[feature_name] = []
                    features[feature_name].append(sentence_features[feature_name])

            for feature_name in features:
                features[feature_name] = torch.cat(features[feature_name]).to(self.device)

            with torch.no_grad():
                out_features = self.forward(features)
                embeddings = out_features['sentence_embedding']
                emded_sequence = out_features['token_embeddings']

                input_mask = out_features['attention_mask']
                input_mask_expanded = input_mask.unsqueeze(-1).expand(emded_sequence.size()).float()
                embed_sequence = emded_sequence * input_mask_expanded

                if convert_to_numpy and not convert_to_tensor:
                    embeddings = np.copy(embeddings.cpu().detach().numpy())
                    embed_sequence = np.copy(embed_sequence.cpu().detach().numpy())

                all_embeddings.extend(embeddings)
                all_embed_sequences.extend(embed_sequence)

        reverting_order = np.argsort(length_sorted_idx)
        all_embeddings = [all_embeddings[idx] for idx in reverting_order]
        all_embed_sequences = [all_embed_sequences[idx] for idx in reverting_order]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
            all_embed_sequences = torch.stack(all_embed_sequences)

        return all_embeddings, all_embed_sequences


class SentenceEncoderService():
    """
    Сервер энкодера
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.featurizer = SentenceTransformerExtended(NLU_ENCODER_CONFIG['featurizer_model'], device="cuda")
        self.max_length = NLU_ENCODER_CONFIG['max_seq_length']
        self.featurizer._modules['0'].max_seq_length = self.max_length

    def encode(self, inputs, **kwargs):
        return np.array(self.featurizer.encode(inputs, **kwargs))

    def decode(self, inputs, **kwargs):
        return self.featurizer.decode(inputs, **kwargs)

    def tokenize(self, inputs):
        return np.array(self.featurizer.tokenize(inputs))

    def tokenize_dataset(self, text_sequence):
        return self.featurizer.tokenize(text_sequence)

    def featurize(self, list_inputs, convert_to_numpy=False):
        tokenized = self.featurizer.tokenize(list_inputs)
        pooled_out, encoded_seq = self.featurizer.featurize(tokenized, convert_to_numpy=convert_to_numpy,
                                                            is_pretokenized=True)
        # pooled_out, encoded_seq = list(map(lambda i: tf.constant(i)[None, :], \
        #                             self.featurizer.featurize(seq, convert_to_numpy=True, is_pretokenized=True)))
        encoded_seq = [i.tolist() for i in encoded_seq]
        pooled_out = [i.tolist() for i in pooled_out]
        return {'tokenized': tokenized, 'encoded': encoded_seq, 'embeddings': pooled_out}


class SentenceEncoder:
    """
    Клиент энкодера, адрес указывать в config.json
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.url = NLU_ENCODER_CONFIG["encoder_server"]

    def convert(self, encoder_response):
        # print(encoder_response)
        tokenized = encoder_response['encoded']['tokenized']
        encoded = torch.FloatTensor(encoder_response['encoded']['encoded'])
        embeddings = torch.FloatTensor(encoder_response['encoded']['embeddings'])
        return {'tokenized': tokenized, 'encoded': encoded, 'seq_embeddings': embeddings}

    def encode(self, sequence):
        json_response = json.dumps({"sequence": sequence})
        headers = {'accept': 'application/json', 'content-type': 'application/json'}
        response = requests.post(self.url + '/encode/', data=json_response, headers=headers)
        response = json.loads(response.text)
        return self.convert(response)


if __name__ == "__main__":
    while True:
        int(input())
        import datetime

        start = datetime.datetime.now()
        print(start)
        st = SentenceEncoder()
        print(st.encode(["wake the fuck up, samurai"])['encoded'])
        print(datetime.datetime.now())
