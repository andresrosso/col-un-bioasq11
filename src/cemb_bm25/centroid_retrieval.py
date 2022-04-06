import re
import copy

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize

def clean(text):
    pattern = r'[0-9]'
    stripped = text.strip().lower() #del
    new_string = re.sub(pattern, '', stripped) #delete numbers
    cleaned_text = re.sub(r'[^a-z0-9\s]','',new_string)
    return cleaned_text

def tokenize(text):
    data = []
    for sentence in sent_tokenize(text):
        temp = []
        for word in word_tokenize(sentence):
            temp.append(word)
        data.extend(temp) #data= cada frase tokenizada... en total tenemos 2593 queries
    return(data)

def subclean_doc(document_attribute, use_tokens):
    if use_tokens:
        return tokenize(clean(document_attribute))
    return clean(document_attribute)

def tokenize_document(document_info, use_tokens):
    return {
        'id': document_info['id'],
        'title': subclean_doc(document_info['title'], use_tokens),
        'abstract': subclean_doc(document_info['abstract'], use_tokens),
        'score': document_info['score']
    }

def extract_unique_doc_info(questions):
    unique_docs = {}
    unique_docs_no_token = {}
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
                unique_docs[document_id] = tokenize_document(
                    document,
                    True
                )
                unique_docs_no_token[document_id] = tokenize_document(
                    document,
                    False
                )
                
                
    return unique_docs, unique_docs_no_token

def select_questions_useful_documents(questions, unique_docs, use_tokens):
    copy_questions = copy.deepcopy(questions)
    valid_docs = set(unique_docs.keys())
    question_solving_doc_ids = set()
    for question in tqdm(copy_questions, desc='Selecting useful documents'):
        question['body'] = subclean_doc(question['body'], use_tokens)
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
        question['documents'] = sorted(
            question['documents'],
            key= lambda x: x['score'],
            reverse=True
        )

def update_question_scores_from_raw_data(raw_questions, question_scores):
    for question in tqdm(raw_questions, desc='Updating dictionary with centroid scores'):
        useful_documents = []
        for document in question['documents']:
            if document['id'] in question_scores[question['id']].keys():
                document['score'] = question_scores[question['id']][document['id']]
                useful_documents.append(
                    document
                )
        useful_documents = sorted(
            useful_documents,
            key= lambda x: x['score'],
            reverse=True
        )
        
        question['documents'] = useful_documents