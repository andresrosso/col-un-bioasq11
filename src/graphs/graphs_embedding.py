import copy
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.spatial as sp

from src.elastic_search_utils.elastic_utils import save_json

def extract_embeddings(texts_x, model_x):
    embeds = []
    for sent in texts_x:
        if not isinstance(model_x, list):
            try:
                wij = model_x.wv[sent] #wij = vector de dimension 200 (wij vec)
            except:
                wij = np.zeros(200) #if the word is not in the model, it creates a 200 dim zero vector
        else:
            try:
                wij = model_x[0].wv[sent]
            except:
                try:
                    wij = model_x[1].wv[sent]
                except:
                    wij = np.zeros(200)
        embeds.append(wij.tolist())
    return(embeds)

def embed_question_and_docs(question, models):
    embed_question = {
        'question_id': question['id'],
        'body': extract_embeddings(question['body'], models['question']),
        'documents': []
    }
    for document in question['documents']:
        document_embeddings = copy.deepcopy(document)
        document_embeddings['entities'] = extract_embeddings(
            document['entities'],
            [models['abstract'], models['title']]
        )
        embed_question['documents'].append(document_embeddings)
    return embed_question

def extract_document_ids(question):
    return [document['id'] for document in question['documents']]

def extract_document_scores(question):
    return [document['score'] for document in question['documents']]

def extract_document_origins(question):
    return [document['origin'] for document in question['documents']]


def extract_document_edges(question, max_entities, similarity_relevance):
    document_edges = []
    for document in question['documents']:
        document_embeddings = document['entities']
        entity_similarity = 1 - sp.distance.cdist(
            document_embeddings, document_embeddings, 'cosine'
        )
        entity_similarity = np.nan_to_num(entity_similarity)  # FIXME
        cropped_matr = entity_similarity[:int(max_entities),:int(max_entities)]
        padding_rows = [0,int(int(max_entities)-cropped_matr.shape[0])]
        padding_cols = [0,int(int(max_entities)-cropped_matr.shape[1])]
        padded_similarity = np.pad(cropped_matr,[padding_rows, padding_cols], mode='constant')
        relevant_edges = [val.tolist() for val in np.where(padded_similarity <= similarity_relevance)]
        document_edges.append(relevant_edges)
    return document_edges

def extract_document_similarities(question, similarity_shape):
    document_similarities = []
    question_embeddings = question['body']
    for document in question['documents']:
        document_embeddings = document['entities']
        question_document_similarity = 1 - sp.distance.cdist(
            document_embeddings, question_embeddings, 'cosine'
        )
        question_document_similarity = np.nan_to_num(question_document_similarity)
        cropped_matr = question_document_similarity[
            :int(similarity_shape[0]),:int(similarity_shape[1])
        ]
        padding_rows = [0,int(int(similarity_shape[0])-cropped_matr.shape[0])]
        padding_cols = [0,int(int(similarity_shape[1])-cropped_matr.shape[1])]
        padded_similarity = np.pad(cropped_matr,[padding_rows, padding_cols], mode='constant')
        document_similarities.append(padded_similarity.tolist())
        
    return document_similarities

def extract_document_labels(question, score_threshold):
    document_labels = []
    for document in question['documents']:
        if (
            (document['origin'] == 'original') or
            (document['score'] >= score_threshold)
        ):
            document_label = 1.0
        else:
            document_label = 0.0
        document_labels.append(document_label)
    return document_labels

def write_question_graphs(question_id, question_graph, saving_path):
    documents_metadata = []
    for document_number, document_id \
        in enumerate(question_graph['document_ids']):
        base_document_info = {
            'question_id': question_id,
            'document_id': document_id,
            'label': question_graph['labels'][document_number],
            'score': question_graph['scores'][document_number],
            'origin': question_graph['origins'][document_number]
        }
        documents_metadata.append(
            base_document_info
        )
        
        document_data = copy.deepcopy(base_document_info)
        document_data.update({
            'similarity_matrix': question_graph['similarity_matrices'][document_number],
            'edges': question_graph['edges'][document_number],
        })
        document_saving_path = f'{saving_path}/{question_id}_{document_id}.json'
        save_json(document_data, document_saving_path)

    return documents_metadata

def make_question_graphs(question, score_threshold, similarity_relevance, similarity_shape, models, saving_path):
    embed_question = embed_question_and_docs(question, models)

    question_graph = {
        'document_ids': extract_document_ids(question),
        'similarity_matrices': extract_document_similarities(embed_question, similarity_shape),
        'edges': extract_document_edges(embed_question, similarity_shape[0], similarity_relevance),
        'labels': extract_document_labels(embed_question, score_threshold),
        'scores': extract_document_scores(embed_question),
        'origins': extract_document_origins(embed_question)
    }

    documents_metadata = write_question_graphs(question['id'], question_graph, saving_path)
    
    return documents_metadata


def make_all_question_graphs(questions, score_threshold, similarity_relevance, similarity_shape, models, saving_path):
    documents_metadata = []
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        
    for question in tqdm(questions['questions'], desc=f'Storing question graphs to {saving_path}'):
        question_metadata = make_question_graphs(
            question,
            score_threshold,
            similarity_relevance,
            similarity_shape,
            models,
            saving_path
        )
        documents_metadata.extend(documents_metadata)
    df_documents_metadata = pd.DataFrame(documents_metadata)
    metadata_path = f'{saving_path}/metadata.parquet'
    df_documents_metadata.to_parquet(metadata_path, index=False)
