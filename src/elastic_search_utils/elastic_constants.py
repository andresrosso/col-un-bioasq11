from enum import Enum
from elasticsearch import Elasticsearch
import sys

sys.path.append('../../')
import globals

class ElasticServer(Enum):
    DEFAULT = Elasticsearch(globals.ES.server) #Elasticsearch('http://localhost:9200')

class TrainingSetPath(Enum):
    TASK9B = '/opt/DR_tests/training9b.json'
    TASK10B = '/opt/DR_tests/training10b.json'

class SearchFields(Enum):
    DEFAULT = globals.ES.search_fields#['title', 'abstract', 'mesh_terms']  # Only functional

class SearchIndex(Enum):
    COMPLETE = globals.ES.index #'pubmed2023-old'
