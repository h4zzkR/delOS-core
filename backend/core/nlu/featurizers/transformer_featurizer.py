import numpy as np
from numpy import ndarray
from typing import List, Dict, Tuple, Iterable, Type, Union
import tensorflow as tf
import torch
from torch import nn, Tensor
from backend.config import NLU_CONFIG
from sentence_transformers import SentenceTransformer, models
from ..tools.utils import encode_dataset

class SentenceTransformerExtended(SentenceTransformer):
    def __init__(self, model_name_or_path: str = None, modules: Iterable[nn.Module] = None, device: str = 'cuda'):
        super().__init__(model_name_or_path, modules, device)

    def decode(self, text):
        return self._first_module().tokenizer.decode(text)

    def featurize(self, sentences: Union[str, List[str], List[int]], batch_size: int = 8, \
                convert_to_numpy: bool = True, convert_to_tensor: bool = False, \
                is_pretokenized: bool = False) -> List[ndarray]:
        self.eval()
        if isinstance(sentences, str): ##Individual sentence
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

class SentenceFeaturizer():
    def __init__(self):
        # self.featurizer = SentenceTransformer(NLU_CONFIG['featurizer_model'])
        self.featurizer = SentenceTransformerExtended(NLU_CONFIG['featurizer_model'])
        self.max_length = NLU_CONFIG['max_seq_length']
        self.featurizer._modules['0'].max_seq_length = self.max_length
    
    def encode(self, inputs, **kwargs):
        return np.array(self.featurizer.encode(inputs, **kwargs))

    def decode(self, inputs, **kwargs):
        return self.featurizer.decode(inputs, **kwargs)

    def tokenize(self, inputs):
        return np.array(self.featurizer.tokenize(inputs))

    def encode_dataset(self, text_sequences):
        token_ids = np.zeros(shape=(len(text_sequences), self.max_length), dtype=np.int32)
        for i, text_sequence in enumerate(text_sequences):
            encoded = self.featurizer.tokenize(text_sequence)
            token_ids[i, 0:len(encoded)] = encoded
        return token_ids

    def featurize(self, inputs):
        seq = self.encode_dataset([inputs]).tolist()
        pooled_out, encoded_seq = list(map(lambda i: tf.constant(i)[None, :], \
                                    self.featurizer.featurize(seq, convert_to_numpy=True, is_pretokenized=True)))
        return seq, encoded_seq, pooled_out