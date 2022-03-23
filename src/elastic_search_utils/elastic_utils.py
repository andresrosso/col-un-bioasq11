import json
import itertools
import logging
from tqdm import tqdm
from src.elastic_search_utils.elastic_constants import (
    ElasticServer,
    TrainingSetPath,
    SearchFields,
    SearchIndex
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from joblib import Parallel, delayed

# READING FUNCTIONS

def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)
    
def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file)

def load_questions(path=TrainingSetPath.TASK9B.value):
    return load_json(path)['questions']
 
def extract_pubmed_id(url):
    return url.split("pubmed/")[1]

def extract_document_ids(question, format_values=True):
    return [
        f"d{extract_pubmed_id(document)}" if format_values
        else extract_pubmed_id(document)
        for document in question['documents']
    ]

def extract_all_document_ids(questions):
    return list(itertools.chain.from_iterable([
        extract_document_ids(question) for question in tqdm(
            questions, desc="Document id extraction: "
        )
    ]))

def extract_formatted_question_id(question):
    question_id = question["id"]
    formatted_question_id = f"q{question_id}"
    return formatted_question_id

def ask_single_doc_id(doc_id='1', fields=SearchFields.DEFAULT.value, es_client=ElasticServer.DEFAULT.value, index=SearchIndex.COMPLETE.value):
    unformatted_doc_id = doc_id.replace('d', '')

    body={"query": {"match": {"pmid" : unformatted_doc_id}}}
    try:
        answer = dict(
            es_client.search(index=index, body=body)
        )['hits']['hits'][0]
    except Exception as e:
        logger.info("Failed document: %s", doc_id)
        return {
            doc_id: 'failed'
        }
    if sorted(fields) == ['abstract', 'mesh_terms', 'title']:
        formatted_answer = {
            doc_id : {
                'score': 1,  # FIXME no idea why there is score in the original answer
                'title': answer['_source']['title'],
                'abstract': answer['_source']['abstract'],
                'mesh_terms': mesh_terms_to_list(
                    answer['_source']['mesh_terms']
                )
            }
        }
    return formatted_answer

def ask_all_doc_id(doc_ids=['1'], fields=SearchFields.DEFAULT.value, es_client=ElasticServer.DEFAULT.value, index=SearchIndex.COMPLETE.value):
    doc_answers = {}
    doc_responses = [
        ask_single_doc_id(
            doc_id,
            fields,
            es_client,
            index
        ) for doc_id in tqdm(doc_ids, desc='Querying training document ids')
    ]
    for doc_response in doc_responses:
        doc_id = list(doc_response.keys())[0]
        doc_answer = doc_response[doc_id]
        doc_answers[doc_id] = doc_answer

    return doc_answers

# Does not work from jupyterhub
def ask_all_doc_id_parallel(doc_ids=['1'], fields=SearchFields.DEFAULT.value, es_client=ElasticServer.DEFAULT.value, index=SearchIndex.COMPLETE.value, n_jobs=5):
    doc_answers = {}

    doc_responses = Parallel(n_jobs=n_jobs)(
        delayed(ask_single_doc_id)(
            doc_id=doc_id,
            fields=fields,
            es_client=es_client,
            index=index
        )
        for doc_id in doc_ids
    )

    for doc_response in doc_responses:
        doc_id = list(doc_response.keys())[0]
        doc_answer = doc_response[doc_id]
        doc_answers[doc_id] = doc_answer

    return doc_answers 

def make_single_gold_standard(question):
    formatted_question_id = extract_formatted_question_id(
        question
    )
    document_ids = extract_document_ids(question)
    gold_standard = {
        formatted_question_id: {
            'question': question['body'],
            'documents': {
                    document_id: 1
                    for document_id in document_ids
            }
        }
    }
    return gold_standard

def make_gold_standard(questions):
    return [
        make_single_gold_standard(question)
        for question in tqdm(questions, desc='Making base gold standard')
    ]

def add_doc_info_to_gold_standard(gold_standard, docs_info):
    for question in tqdm(gold_standard, desc='Adding doc info to gold standard'):
        question_id = list(question.keys())[0]
        for document in question[question_id]['documents'].keys():
            question[question_id]['documents'][document] = \
                docs_info.get(document, 'failed')


def ask_single_question(question='', fields=SearchFields.DEFAULT.value, size=10, es_client=ElasticServer.DEFAULT.value, index=SearchIndex.COMPLETE.value):
    str_question = \
        question['body'] if isinstance(question, dict) else question
    body = {
        "from": 0,
        "size": size,
        "query": {
            "multi_match": {
                "query": str_question,
                "fields": fields
            }
        }
    }
    response = dict(
        es_client.search(
            index=index,
            body=body
        )
    
    )
    return response

def answers_to_id_metric(answers):
    answers_hits = answers['hits']['hits']
    answer_id_metric = {
        f"d{answer['_source']['pmid']}":
            answer['_score'] / 100
        for answer in answers_hits
    }

    return answer_id_metric

def mesh_terms_to_list(mesh_terms):
    return [mesh_term.strip(' ') for mesh_term in mesh_terms.split(';')]

def extract_title_abstract_mesh_terms(answers):
    answers_hits = answers['hits']['hits']
    answer_title_abstract = {
        f"d{answer['_source']['pmid']}": {
            'score': answer['_score'] / 100,
            'title': answer['_source']['title'],
            'abstract': answer['_source']['abstract'],
            'mesh_terms': mesh_terms_to_list(
                answer['_source']['mesh_terms']
            )
        }
        for answer in answers_hits
    }
    return answer_title_abstract

def get_single_question_metrics(question='', fields=SearchFields.DEFAULT.value, size=10, es_client=ElasticServer.DEFAULT.value, index=SearchIndex.COMPLETE.value):
    raw_answer = ask_single_question(
        question,
        fields,
        size,
        es_client,
        index
    )
    if isinstance(question, str):
        return {
            'documents': extract_title_abstract_mesh_terms(
                raw_answer
            )
        }

    question_id = extract_formatted_question_id(
        question
    )
    # TODO other cases
    if sorted(fields) == ['abstract', 'mesh_terms', 'title']:

        single_question_metrics = {
            question_id: {
                'question': question['body'],
                'documents': extract_title_abstract_mesh_terms(
                    raw_answer
                )
            }
        }
        
        
    return single_question_metrics 

def ask_several_questions(questions=[], fields=SearchFields.DEFAULT.value, size=10, es_client=ElasticServer.DEFAULT.value, index=SearchIndex.COMPLETE.value):
    metric_answers = []
    for question in tqdm(questions, desc="Extracting docs from elastic search"):
        question_metrics = get_single_question_metrics(
            question,
            fields,
            size,
            es_client,
            index
        )
        metric_answers.append(question_metrics)
    return metric_answers
