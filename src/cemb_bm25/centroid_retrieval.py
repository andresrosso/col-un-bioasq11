import re
import copy
import itertools

import gensim
from gensim.models import Word2Vec

import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import en_core_sci_sm

from joblib import Parallel, delayed

from src.cemb_bm25.text_cleanup import standardize_text, clean_sentence

ENTITY_EXTRACTOR = en_core_sci_sm.load()

def tokenize(text):
    data = []
    standardized_text = standardize_text(text)
    for sentence in sent_tokenize(standardized_text):
        cleaned_sentence = clean_sentence(sentence)
        temp = []
        for word in word_tokenize(cleaned_sentence):
            temp.append(word)
        data.extend(temp) #data= cada frase tokenizada... en total tenemos 2593 queries
    return(data)

def extract_entities(text):
    data = []
    standardized_text = standardize_text(text)
    entities = []
    for sentence in sent_tokenize(standardized_text):
        cleaned_sentence = clean_sentence(sentence)
        raw_entities = ENTITY_EXTRACTOR(cleaned_sentence).ents
        sentence_entities = list(
            itertools.chain.from_iterable([
                entity.text.split() for entity in raw_entities
            ])
        )
        entities.extend(sentence_entities)
    return entities

def tokenize_document(document_info):
    return {
        'id': document_info['id'],
        'title': tokenize(document_info['title']),
        'abstract': tokenize(document_info['abstract']),
        'score': document_info['score']
    }

def extract_document_entities(document_info):
    return {
        'id': document_info['id'],
        'entities': (
            extract_entities(document_info['title']) +
            extract_entities(document_info['abstract'])
        ),
        'score': document_info['score']
    }

def extract_unique_doc_info(questions):
    unique_docs = {}
    for question in tqdm(questions, desc='Extracting unique doc info'):
        documents = question['documents']
        for document in documents:
            document_id = document['id']
            if (
                unique_docs.get(document_id, False) or
                (document['abstract'] == '') or
                (document['abstract'] is None)
            ):
                continue
            else:
                unique_docs[document_id] = document
    return unique_docs

def docs_to_tokens(unique_docs, n_jobs=5, verbose=10):
    doc_ids = list(unique_docs.keys())
    doc_tokens = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(tokenize_document)(doc_info)
        for doc_info in unique_docs.values()
    )
    
    docs_tokens = dict(zip(doc_ids, doc_tokens))

    return docs_tokens

def docs_to_entities(unique_docs, n_jobs=5, verbose=10):
    doc_ids = list(unique_docs.keys())
    doc_entities = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(extract_document_entities)(doc_info)
        for doc_info in unique_docs.values()
    )

    docs_entities = dict(zip(doc_ids, doc_entities))

    return docs_entities

def extract_unique_doc_entity_info(questions):
    unique_docs = {}
    for question in tqdm(questions, desc='Extracting unique doc entities'):
        documents = question['documents']
        for document in documents:
            document_id = document['id']
            if (
                unique_docs.get(document_id, False) or
                (document['abstract'] == '') or
                (document['abstract'] is None)
            ):
                continue
            else:
                unique_docs[document_id] = extract_document_entities(
                    document
                )
    return unique_docs

def select_questions_useful_documents(questions, unique_docs):
    copy_questions = copy.deepcopy(questions)
    valid_docs = set(unique_docs.keys())
    question_solving_doc_ids = set()
    for question in tqdm(copy_questions, desc='Selecting useful documents'):
        question['body'] = tokenize(question['body'])
        new_documents = []
        new_document_ids = set()
        for document in question['documents']:
            document_id = document['id']
            if document_id in valid_docs:
                new_documents.append(unique_docs[document_id])
                new_document_ids.add(document_id)
        question['documents'] = new_documents
        question_solving_doc_ids = question_solving_doc_ids | set(new_document_ids)
    return copy_questions, list(question_solving_doc_ids)

def extract_unique_titles_and_abstracts(tokenized_unique_docs, question_solving_doc_ids):
    df_tokenized_doc_info = pd.DataFrame(tokenized_unique_docs).transpose()
    df_tokenized_doc_info.drop_duplicates(
        subset=['id'],
        inplace=True
    )
    not_question_solving_index = df_tokenized_doc_info[
        ~df_tokenized_doc_info['id'].isin(question_solving_doc_ids)
    ].index
    
    df_tokenized_doc_info.drop(
        not_question_solving_index,
        inplace=True
    )
    unique_abstract_tokens = df_tokenized_doc_info['abstract'].values
    unique_title_tokens = df_tokenized_doc_info['title'].values
    document_ids = df_tokenized_doc_info['id'].values
    
    abstract_tokens = dict(zip(document_ids, unique_abstract_tokens))

    title_tokens = dict(zip(document_ids, unique_title_tokens))

    return abstract_tokens, title_tokens

def extract_unique_questions(tokenized_questions):
    unique_questions = {
        question['id']: question['body']
        for question in tokenized_questions
    }
    return unique_questions

def fit_bio_w2vec(tokens_text):
    model_bio2vec = gensim.models.word2vec.Word2Vec(window = 5,vector_size=200, min_count = 1, workers = 3,sg= 0)
    # Build the Volabulary for titles
    model_bio2vec.build_vocab(tokens_text)
    vocab = model_bio2vec.wv.key_to_index
    #Train the Doc2Vec model
    model_bio2vec.train(tokens_text, total_examples=model_bio2vec.corpus_count, epochs=model_bio2vec.epochs)
    print('Word2Vec vocabulary length:', len(vocab))
    return(model_bio2vec, vocab)

def calculate_centroids(text_tokens, model, vocab):
    num = 0
    centroids = {}
    for sent_id, sent in tqdm(text_tokens.items(), desc='Extracting centroids'):
        num = []
        for tok in sent:
            if tok in vocab:
                wij = model.wv[tok] #wij = vector de dimension 200 (wij vec)
                num +=[wij] # num = create 200xn matrix, where n is the number of tokens in text
            else: 
                wij = np.zeros(200)
                num+=[wij]
        cent = np.mean(num,axis=0)
        centroids[sent_id] = cent
    return(centroids)

def calculate_cosine_similarity(question_centroid, document_centroid):
    projection = np.dot(question_centroid, document_centroid)
    normalization = np.linalg.norm(question_centroid) * np.linalg.norm(document_centroid)
    cosine_similarity = projection / normalization
    return cosine_similarity

def calculate_question_answer_similarity(tokenized_questions, question_centroids, abstract_centroids, title_centroids):
    questions_similarities = {
        'questions': []
    }
    for question in tqdm(tokenized_questions, desc='Calculating cosine similarity'):
        question_id = question['id']
        question_centroid = question_centroids[question_id]

        question_document_distances = []
        for document in question['documents']:
            document_id = document['id']
            document_score = document['score']

            abstract_centroid = abstract_centroids[document_id]
            title_centroid = title_centroids[document_id]

            abstract_cosine_similarity = calculate_cosine_similarity(
                question_centroid, abstract_centroid
            )
            
            title_cosine_similarity = calculate_cosine_similarity(
                question_centroid, title_centroid
            )
            
            document_distances = {
                'abstract_cosine_similarity': abstract_cosine_similarity,
                'title_cosine_similarity': title_cosine_similarity,
                'id': document_id,
                'score': document_score
            }
            question_document_distances.append(document_distances)
        
        question_similarity = {
            'id': question_id,
            'documents': question_document_distances
        }
        questions_similarities['questions'].append(question_similarity)
    
    return questions_similarities

def calculate_document_centroid_score(document_info, abstract_weight, title_weight):
    centroid_score = document_info['score'] * (
        (abstract_weight * document_info['abstract_cosine_similarity']) +
        (title_weight * document_info['abstract_cosine_similarity'])
    )
    return centroid_score
    
def calculate_centroid_score(questions_similarities, abstract_weight, title_weight):
    centroid_scores = {} 
    for question in tqdm(questions_similarities, desc='Calculating centroid distance'):
        document_scores = {}
        for document in question['documents']:
            document_score = calculate_document_centroid_score(
                document,
                abstract_weight,
                title_weight
            )
            document_scores[document['id']] = document_score
        centroid_scores[question['id']] = document_scores

    return centroid_scores

def update_question_scores(raw_questions, question_scores):
    for question in tqdm(raw_questions, desc='Updating dictionary with centroid scores'):
        for document in question['documents']:
            document['score'] = question_scores[question['id']][document['id']]
            if 'documents_origin' in question.keys():
                document['origin'] = question['documents_origin'][document['id']]
        question['documents'] = sorted(
            question['documents'],
            key= lambda x: x['score'] + 
                (10.0 if x.get('origin', 'queried') == 'original' else 0.0),
            reverse=True
        )
        question.pop('documents_origin', None)


def update_question_scores_from_raw_data(raw_questions, question_scores):
    for question in tqdm(raw_questions, desc='Updating dictionary with centroid scores'):
        useful_documents = []
        for document in question['documents']:
            if document['id'] in question_scores[question['id']].keys():
                document['score'] = question_scores[question['id']][document['id']]
                if 'documents_origin' in question.keys():
                    document['origin'] = question['documents_origin'][document['id']]
                else:
                    document['origin'] = 'queried'
                useful_documents.append(
                    document
                )
        useful_documents = sorted(
            useful_documents,
            key= lambda x: x['score'] + 
                (10.0 if x.get('origin', 'queried') == 'original' else 0.0),
            reverse=True
        )
        question['documents'] = useful_documents
        question.pop('documents_origin', None)

def load_bio_w2vec_model(path):
    return Word2Vec.load(path)

def calculate_centroids_test(text_tokens, model):  # FIXME repeated code of calculate_centroids but no vocab
    num = 0
    centroids = {}
    for sent_id, sent in tqdm(text_tokens.items(), desc='Extracting centroids'):
        num = []
        for tok in sent:
            try:
                wij = model.wv[tok] #wij = vector de dimension 200 (wij vec)
                num +=[wij] # num = create 200xn matrix, where n is the number of tokens in text
            except: 
                wij = np.zeros(200)
                num+=[wij]
        cent = np.mean(num,axis=0)
        centroids[sent_id] = cent
    return(centroids)

# MERGE METHODS

def merge_origins(original_questions, queried_questions):
    failed = 0
    merged_questions = copy.deepcopy(original_questions)
    for question_index, original_question in tqdm(enumerate(merged_questions), desc='merging dictionaries'):
        queried_question = queried_questions[question_index]
        
        original_documents = \
            copy.deepcopy(original_question['documents'])
        original_document_ids = set([
            original_document['id']
            for original_document in original_documents
        ])
        non_repeating_queried_documents = [
            copy.deepcopy(document)
            for document in queried_question['documents']
            if document['id'] not in original_document_ids
        ]
        non_repeating_queried_documents_ids = [
            document['id'] for document
            in non_repeating_queried_documents
        ]

        merged_documents = list(itertools.chain(
            original_documents,
            non_repeating_queried_documents
        ))
        original_question['documents'] = merged_documents
        
        document_origin_map = {
            doc_id: 'original' for doc_id in original_document_ids
        }
        
        for doc_id in non_repeating_queried_documents_ids:
            document_origin_map[doc_id] = 'queried'

        original_question['documents_origin'] = document_origin_map
        
    return {'questions': merged_questions}
