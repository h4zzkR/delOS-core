import tensorflow as tf
from pathlib import Path
from ..intent_classificator.model import RequestIntentClassifier
from ..nlu_data.utils import DatasetLoader, space_punct
from transformers import TFBertModel, BertTokenizer

class NLU():
    def __init__(self, model_name="bert-base-cased"):
        self.curdir = Path(__file__).parent.absolute()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        bertbase = TFBertModel.from_pretrained(model_name)
        d = DatasetLoader('merged') # TODO add here dataset param or join all dsets
        self.intent2id, self.id2intent = d.load_intents_map()

        self.classifier = RequestIntentClassifier(bertbase, self.id2intent)
        self.tagger = None

    def __call__(self, text):
        text = space_punct(text)
        inputs = tf.constant(self.tokenizer.encode(text))[None, :]  # batch_size = 1

        intent = self.classifier.classify(inputs)

        # slot_logits, intent_logits = self.model(inputs)
        # slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
        intent_id = intent_logits.numpy().argmax(axis=-1)[0]

        return self.decode_predictions(text, intent, slot_ids)

    def decode_predictions(self, text, intent_id, slot_ids):
        # TODO intent_id to intent
        """
        Model output to json-like data
        {'intent' : name, 'slots' : {'a' : 'b'}}
        """
        info = {"intent": self.id2intent[intent_id]}
        collected_slots = {}
        active_slot_words = []
        active_slot_name = None
        for word in text.split():
            tokens = self.tokenizer.tokenize(word)
            current_word_slot_ids = slot_ids[:len(tokens)]
            slot_ids = slot_ids[len(tokens):]
            current_word_slot_name = self.id2slot[current_word_slot_ids[0]]
            if current_word_slot_name == "O":
                if active_slot_name:
                    collected_slots[active_slot_name] = " ".join(active_slot_words)
                    active_slot_words = []
                    active_slot_name = None
            else:
                # Naive BIO: handling: treat B- and I- the same...
                new_slot_name = current_word_slot_name[2:]
                if active_slot_name is None:
                    active_slot_words.append(word)
                    active_slot_name = new_slot_name
                elif new_slot_name == active_slot_name:
                    active_slot_words.append(word)
                else:
                    collected_slots[active_slot_name] = " ".join(active_slot_words)
                    active_slot_words = [word]
                    active_slot_name = new_slot_name
        if active_slot_name:
            collected_slots[active_slot_name] = " ".join(active_slot_words)
        info["slots"] = collected_slots
        return info


if __name__ == "__main__":
    nlu = NLU()
    inp = input()
    print(nlu(inp))