import re
from qa_data import QAPair
import socket
from elasticsearch import Elasticsearch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from operator import itemgetter
import nltk
import nltk.data
import requests
import os

#load from enviroment
es_host = '127.0.0.1:9200'
if 'ES_HOST' in os.environ:
    es_host = os.environ.get('ES_HOST')
    
es = Elasticsearch(hosts=[es_host])
stop_words = set(stopwords.words('english'))
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def reload():
    es = Elasticsearch(hosts=[es_host])
    
    
def search_docs(question, index_name, num_docs):
    doc_list = []
    try:
        query = {
            "from" : 0, 
            "size" : 10,
            "query": {
                "multi_match" : {
                      "query":    "replace it!", 
                      "fields": [ "abstract", "title" ] 
                    }
            }
        }
        query['size'] = num_docs
        query['query']['multi_match']['query']=question
        res = es.search(index=index_name, body=query)
        for doc in res['hits']['hits']:
            #print("%s) %s" % (doc['_id'], doc['_source']))
            #print(doc['_id'],doc['_source']['title'])
            doc_list += [(doc['_id'],doc['_source']['title'],doc['_source']['abstract'])]
    except Exception as e:
        print(("Error in query: ",question))
        print(e)
    return doc_list

def search_docs_mesh(question, index_name, num_docs):
    doc_list = []
    try:
        query = {
            "from" : 0, 
            "size" : num_docs,
            "query": {
                "multi_match" : {
                      "query":    question, 
                      "type":     "cross_fields",
                      "fields": [ "abstract", "title", "mesh" ],
                      "operator": "or"
                    }
            }
        }
        res = es.search(index=index_name, body=query, request_timeout=30)
        for doc in res['hits']['hits']:
            doc_list += [(doc['_id'],doc['_source']['title'],doc['_source']['abstract'],doc['_source']['mesh'])]
    except Exception as e:
        print(("Error in query: ",question))
        print(e)
    return doc_list

def get_doc(doc_id, index_name):
    try:
        query = {
            "from" : 0, "size" : 1,
            "query": {
                "multi_match" : {
                      "query":    "replace it!", 
                      "fields": [ "_id" ] 
                    }
            }
        }
        query['query']['multi_match']['query']=doc_id
        res = es.search(index=index_name, body=query)
        doc = None
        for doc in res['hits']['hits']:
            #print("%s) %s" % (doc['_id'], doc['_source']))
            #print(doc['_id'],doc['_source']['title'])
            doc = (doc['_id'],doc['_source']['title'],doc['_source']['abstract'])
    except Exception as e:
        print(("Error in query: ",doc_id))
        print(e)
    return doc


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def build_cuitext_pairs(q, q_id, doc, only_text=False):
    #[QAPair(data['id'], data['question'], str(index_ans), ans, 0)]
    q_list = []
    for i_s, sentence in enumerate(doc):
        if only_text:
            q_list += [QAPair(q_id, (q, []), str(q_id)+'_'+str(i_s),
                          (sentence,[]), 0)]
        else:
            q_list += [QAPair(q_id, (q, cui_extract_concepts(q)), str(q_id)+'_'+str(i_s),
                      (sentence,cui_extract_concepts(sentence)), 0)]
    return q_list


def build_pairs(q, q_id, doc):
    #[QAPair(data['id'], data['question'], str(index_ans), ans, 0)]
    q_list = []
    for i_s, sentence in enumerate(doc):
        q_list += [QAPair(q_id, q, str(q_id)+'_'+str(i_s), sentence, 0)]
    return q_list

def cui_extract_concepts(text, verbose=0):
    query = {"text": text }
    resp = requests.post('http://localhost:5000/match', json=query)
    cui_concepts = []
    if resp.status_code != 200:
        # This means something went wrong.
        raise ApiError('GET /tasks/ {}'.format(resp.status_code))
    for todo_item in resp.json():
        cui_concepts += [todo_item['cui']]
        if verbose > 0:
            print(('{} {} {}'.format(todo_item['term'], todo_item['cui'], todo_item['similarity'])))        
    return cui_concepts
