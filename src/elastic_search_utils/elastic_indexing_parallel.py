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
from joblib import Parallel, delayed
from multiprocessing import Pool, Process, Manager


index = 'pubmed2023-old'
es = Elasticsearch('http://localhost:9200')
path = '/opt/bioasq/resources/pubmed_baseline_2023/'

dirs = os.listdir( path )
read_dirs = [f"{path}{dirx}" for dirx in dirs]
failures = {}
n_file = 0

def process_file(file):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parser for BIOASQ'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        required=True
    )
    args = parser.parse_args()
    PREFIX = args.prefix
    
    print(f"USED PREFIX: {PREFIX}")
    
    filter_read_dirs = [
        read_dir for read_dir in read_dirs
        if PREFIX in read_dir
    ]
    print(f"INDEX NAME: {index}")
    print(f"KEPT {len(filter_read_dirs)} FILES FROM THE ORIGINAL {len(read_dirs)}")
    print(f"EXAMPLE FILE: {filter_read_dirs[0]}")

    # NOT DOING THIS RIGHT NOW
    try:
        es.indices.delete(index=index)
        print('Deleted index, fix')
    except:
        print('Not found index')
    print('CREATING')
    es.indices.create(index=index)

    print(f"INDEXING {len(filter_read_dirs)} FILES")
    
    #Parallel(n_jobs=10)(delayed(process_file)(file) for file in filter_read_dirs )

    pool = Pool(8)
    df_collection = pool.map(process_file, filter_read_dirs)
    pool.close()
    pool.join()
    
    with open(f'{PREFIX}_failures.json', 'w') as fails_file:
        json.dump(failures, fails_file)

