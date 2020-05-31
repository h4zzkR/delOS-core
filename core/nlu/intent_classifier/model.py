import os
import tensorflow as tf
from transformers import BertTokenizer
from pathlib import Path
from .base import RequestIntentClassifier as Classifier


class RequestIntentClassifier():
    def __init__(self, model_name, id2intent, load=True):

        curdir = Path(__file__).parent.absolute()
        self.id2intent = id2intent
        intents_number = len(self.id2intent.keys())

        self.model = Classifier(model_name, intents_number)

        ckp_path = self.check_model(curdir)
        self.model.load_weights(ckp_path)


    def check_model(self, curdir):
        p = os.path.join(curdir, 'model/')
        ckp = Path(os.path.join(p, 'checkpoint')).read_text()
        m = ckp.splitlines()[0].split(': ')[-1]
        p = os.path.join(p, m[1:-1])
        return p

    def classify(self, inputs):
        class_id = self.model(inputs).numpy().argmax(axis=1)[0]
        return class_id
        # return self.id2intent[class_id]



class SlotIntentDetectorModel():

    def __init__(self, model_name="bert-base-cased", load=True):
        curdir = Path(__file__).parent.absolute()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        intents_names, intents_map = load_intents_map(curdir)
        slot_names, slot_map = load_slots_map(curdir)

        self.id2intent = {intents_map[i] : i for i in intents_map.keys()}
        self.id2slot = {slot_map[i] : i for i in slot_map.keys()}

        self.model = SlotIntentDetectorModelBase(
            len(self.id2intent.keys()), len(self.id2slot.keys())
            )

        ckp_path = self.check_model(curdir)
        self.model.load_weights(ckp_path)

    def check_model(self, curdir):
        p = os.path.join(curdir, 'model/')
        ckp = Path(os.path.join(p, 'checkpoint')).read_text()
        m = ckp.splitlines()[0].split(': ')[-1]
        p = os.path.join(p, m[1:-1])
        return p

    def decode_predictions(self, text, intent_id, slot_ids):
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

    def classify(self, text, map_intent=True):
        # deprecated
        inputs = tf.constant(self.tokenizer.encode(text))[None, :]  # batch_size = 1
        class_id = self.model(inputs).numpy().argmax(axis=1)[0]
        # print(class_id, self.id2intent)
        return self.id2intent[class_id]


    def nlu(self, text):
        text = space_punct(text)
        inputs = tf.constant(self.tokenizer.encode(text))[None, :]  # batch_size = 1
        slot_logits, intent_logits = self.model(inputs)
        slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
        intent_id = intent_logits.numpy().argmax(axis=-1)[0]

        return self.decode_predictions(text, intent_id, slot_ids)

    def encode(self, text_sequence, max_length):
        return encode_dataset(self.tokenizer, text_sequence, max_length)


if __name__ == "__main__":
    model = RequestIntentClassifier()
    inp = input()
    print(model.classify(inp))