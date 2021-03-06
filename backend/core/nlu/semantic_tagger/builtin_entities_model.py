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
        tags = self.translate_entity(self.parser.parse(item))
        tags_dict = dict()
        for t in tags:
            tag_name = t['entity_kind']
            del t['entity_kind'], t['range']
            tags_dict.update({tag_name : t})
        return tags_dict
    
    def get_all_grammar_entities(self):
        return list(self.entities.values())