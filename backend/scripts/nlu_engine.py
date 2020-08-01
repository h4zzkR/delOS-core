from core.nlu.engine.base_engine2 import NLUEngine

if __name__ == "__main__":
    text = 'turn on the light please'
    n = NLUEngine()
    n.process(text)