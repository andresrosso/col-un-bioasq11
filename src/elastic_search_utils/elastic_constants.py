from enum import Enum
from elasticsearch import Elasticsearch

class ElasticServer(Enum):
    DEFAULT = Elasticsearch('http://localhost:9200')

class TrainingSetPath(Enum):
    TASK9B = '/home/azuluagac/DR_tests/training9b.json'
    TASK10B = '/home/azuluagac/DR_tests/training10b.json'

class SearchFields(Enum):
    DEFAULT = ['title', 'abstract', 'mesh_terms']  # Only functional

class SearchIndex(Enum):
    INCOMPLETE = 'pubmed2022_index'
    COMPLETE = 'pubmed2022_index_full'
