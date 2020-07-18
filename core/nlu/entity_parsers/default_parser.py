from snips_nlu_parsers import BuiltinEntityParser
import json

class RustNERParser():
    """
    Super fast Snips entity parser for some default entities
    for an extra step above others parsers, if tagger says that
    there is snips/ tag in sequence
    """
    def __init__(self):
        self.parser = BuiltinEntityParser.build(language="en")
    
    def parse(self, text, pretty=False):
        parsing = self.parser.parse(text)
        if pretty:
            return json.dumps(parsing, indent=2)
        else:
            return json.dumps(parsing)

    def get_all_kinds(self):
        return ['snips/amountOfMoney', 'snips/duration', 'snips/number',
                'snips/ordinal', 'snips/temperature', 'snips/datetime',
                'snips/date', 'snips/time', 'snips/datePeriod', 'snips/timePeriod',
                'snips/percentage',]

if __name__ == "__main__":
    p = RustNERParser()
    print(p.parse('2019'))