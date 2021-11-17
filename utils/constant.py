"""
Define constants.
"""
MAX_LEN = 100
# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

BIO_TO_ID = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'IN': 3, 'NN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, '.': 10, 'CC': 11, 'CD': 12, 'RB': 13, 'TO': 14, 'VBN': 15, 'VB': 16, 'VBZ': 17, 'VBG': 18, 'PRP': 19, "''": 20, 'POS': 21, 'PRP$': 22, ':': 23, 'VBP': 24, 'MD': 25, 'WP': 26, 'WDT': 27, 'WRB': 28, 'NNPS': 29, 'RP': 30, 'JJS': 31, 'JJR': 32, '$': 33, 'RBR': 34, 'EX': 35, 'RBS': 36, '(': 37, ')': 38, 'WP$': 39, '``': 40, 'PDT': 41, 'FW': 42, 'UH': 43, '#': 44, 'SYM': 45}

ENTYPE_TO_ID = {'O': 0, 'STATE_OR_PROVINCE': 1, 'LOCATION': 2, 'CITY': 3, 'ORGANIZATION': 4, 'DATE': 5, 'PERSON': 6, 'TITLE': 7, 'ORDINAL': 8, 'NUMBER': 9, 'COUNTRY': 10, 'DURATION': 11, 'CAUSE_OF_DEATH': 12, 'IDEOLOGY': 13, 'NATIONALITY': 14, 'RELIGION': 15, 'CRIMINAL_CHARGE': 16, 'MISC': 17, 'MONEY': 18, 'URL': 19, 'PERCENT': 20, 'TIME': 21, 'SET': 22}

# nyt
LABEL_TO_ID = {
 "/business/company/founders": 1, 
 "/people/person/place_of_birth": 2, 
 "/people/deceased_person/place_of_death": 3, 
 "/business/company_shareholder/major_shareholder_of": 4, 
 "/people/ethnicity/people": 5, 
 "/location/neighborhood/neighborhood_of": 6, 
 "/sports/sports_team/location": 7, 
 "/business/company/industry": 9, 
 "/business/company/place_founded": 10, 
 "/location/administrative_division/country": 11, 
 "None": 0, 
 "/sports/sports_team_location/teams": 12, 
 "/people/person/nationality": 13, 
 "/people/person/religion": 14, 
 "/business/company/advisors": 15, 
 "/people/person/ethnicity": 16, 
 "/people/ethnicity/geographic_distribution": 17, 
 "/business/person/company": 8, 
 "/business/company/major_shareholders": 19, 
 "/people/person/place_lived": 18, 
 "/people/person/profession": 20, 
 "/location/country/capital": 21, 
 "/location/location/contains": 22, 
 "/location/country/administrative_divisions": 23, 
 "/people/person/children": 24
}
