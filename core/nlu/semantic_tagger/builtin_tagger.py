from snips_nlu_parsers import BuiltinEntityParser

class BuiltinEntityTagger():
    """
    Huge respect to snips-nlu team for
    https://github.com/snipsco/snips-nlu-ontology
    """
    def __init__(self, lang='en'):
        self.parser = BuiltinEntityParser.build(language="en")
        self.entities = {
        'snips/amountOfMoney' : '@amountOfMoney','snips/date' : '@date', \
            'snips/datePeriod' : '@datePeriod','snips/datetime' : '@datetime', \
            'snips/duration' : '@duration', 'snips/number' : '@number', \
            'snips/ordinal' : '@ordinal', 'snips/percentage' : '@percentage', \
            'snips/temperature' : '@temperature', 'snips/time' : '@time', \
            'snips/timePeriod' : '@timePeriod'
        }
        
    def translate_entity(self, items):
        for i in range(len(items)):
            items[i]['entity_kind'] = self.entities[items[i]['entity_kind']]
        return items
    
    def tag(self, item, **kwargs):
        return self.translate_entity(self.parser.parse(item))
    
    def get_all_grammar_entities(self):
        return list(self.entities.values())