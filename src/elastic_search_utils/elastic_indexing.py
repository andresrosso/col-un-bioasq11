import json
import time
from collections import deque
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import os, sys
import gzip
import json
import os, sys
import pubmed_parser as pp
from pprint import pprint
import logging

index = 'pubmed2022'
es = Elasticsearch('http://localhost:9200')
path = '/datasets/pubmed2022/'

dirs = os.listdir( path )
read_dirs = [f"{path}{dirx}" for dirx in dirs]

try:
    es.indices.delete(index=index)
    print('Deleted index')
except:
    print('Not found index')
#es.indices.delete(index='pubmed_2022_index')
print('CREATING')
es.indices.create(index=index)

failures = {}

print(f"INDEXING {len(read_dirs)} FILES")
for n_file, file in enumerate(read_dirs):
    print('\n' * 3)
    print('*' * 20)
    print(f"WRITING {file}, {n_file + 1}/{len(read_dirs)}")
    try:
        f = gzip.open(file, 'rb')
        file_content = f.read()
        dict_out = pp.parse_medline_xml(file_content)
        dict_form = [
            {
                "_id": dictx['pmid'],
                "_source": dictx
            }
            for dictx in dict_out
        ]
        resp = helpers.parallel_bulk(es, dict_form, index=index, chunk_size=2000)
        deque(resp, maxlen=0)
    except Exception as e:
        print(f"FAILED FILE: {file} ({len(failures.keys())} files failed)")
        logging.exception(e)
        failures[file] = str(e)

with open('failures.json', 'w') as fails_file:
    json.dump(failures, fails_file)
