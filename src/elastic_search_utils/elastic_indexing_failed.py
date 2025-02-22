import json
import time
import argparse
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

index = 'pubmed2023-old'
es = Elasticsearch('http://localhost:9200')
path =  '/opt/bioasq/resources/pubmed_baseline_2023/'
n_trials = 3

dirs = os.listdir( path )
read_dirs = [f"{path}{dirx}" for dirx in dirs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parser for Bavaria training MLOps'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        required=True
    )
    args = parser.parse_args()
    PREFIX = args.prefix
    
    print(f"USED PREFIX: {PREFIX}")
    
    with open(f'{PREFIX}_failures.json', 'r') as fails_file:
        failed_files = json.load(fails_file)

    filter_read_dirs = list(failed_files.keys())

    if len(filter_read_dirs) > 0:
        failures = {}

        print(f"INDEXING {len(filter_read_dirs)} FAILED FILES")
        for n_file, file in tqdm(enumerate(filter_read_dirs)):
            print('\n' * 3)
            print('*' * 20)
            print(f"WRITING {file}, {n_file + 1}/{len(filter_read_dirs)}")
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
                resp = helpers.bulk(es, dict_form, index=index)
            except Exception as e:
                print(f"FAILED FILE: {file} ({len(failures.keys())} files failed)")
                logging.exception(e)
                failures[file] = str(e)

        with open(f'{PREFIX}_failures.json', 'w') as fails_file:
            json.dump(failures, fails_file)
    else:
        print("PREFIX FAILURES WERE ALREADY FIXED")
