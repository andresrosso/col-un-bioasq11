import numpy as np
import random
from os.path import exists
from os import makedirs
import os
from random import shuffle
import json

class QAPair():
    def __init__(self, qi, q, ai, a, l):
        self.qi = qi
        self.q = q
        self.ai = ai
        self.a = a
        self.l = l

    def __repr__(self):
        return 'qi('+str(self.qi)+') '+'ai('+str(self.ai)+') '+str(self.l)

class QADataSet(object):

    def __init__(self, name):
        self.name = name
        self.patitions = []
        self.questions = {}

    def get_stats(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def build_qa_pairs(self, dataset):
        raise NotImplementedError("Subclass must implement abstract method")


    def get_random_samples(self, dataset, samples, positive_rate=0.5):
        num_pos_samples = int(samples*(positive_rate))
        positiveSamples = [ q for q in dataset if q.l==1 ]
        negativeSamples = [ q for q in dataset if q.l==0 ]
        data = random.sample(positiveSamples, num_pos_samples)+random.sample(negativeSamples, samples-num_pos_samples)
        shuffle(data)
        return data


#BioASQ 2018
class BiosqDataSet(QADataSet):

    def __init__(self,year,path):
        QADataSet.__init__(self,'BiosqDataSet')
        print((year,path))
        questions = []
        self.question_files = []
        for f in os.listdir(path):
            self.question_files += [path+'/'+f]

    def get_stats(self):
        return 'Number of pairs: '+str(len(self.question_files))

    '''
    Return a tuple of (quesion_id, question, answer_id, answer, label)
    '''
    def build_qa_pairs(self, dataset):
        #Construct Question Answer Pairs
        self.questions_answer_pairs = []
        for f_q in self.question_files:
            data = json.load(open(f_q))
            index_ans = 0
            for ans in data['pos_answers']:
                index_ans += 1
                if(len(ans)>3 and len(data['question'])>3):
                    self.questions_answer_pairs += [QAPair(data['id'], data['question'], str(index_ans), ans, 1)]
            for ans in data['neg_answers']:
                index_ans += 1
                if(len(ans)>3 and len(data['question'])>3):
                    self.questions_answer_pairs += [QAPair(data['id'], data['question'], str(index_ans), ans, 0)]
        return self.questions_answer_pairs
    
#BioASQ 2019
class BioasqCuiTextDataSet(QADataSet):

    def __init__(self,year,path):
        QADataSet.__init__(self,'BioasqCuiTextDataSet')
        print((year,path))
        questions = []
        self.question_files = []
        for f in os.listdir(path):
            self.question_files += [path+'/'+f]

    def get_stats(self):
        return 'Number of pairs: '+str(len(self.question_files))

    '''
    Return a tuple of (quesion_id, question, quiestion_cui answer_id, answer, label)
    '''
    def build_qa_pairs(self, dataset):
        #Construct Question Answer Pairs
        self.questions_answer_pairs = []
        for f_q in self.question_files:
            data = json.load(open(f_q))
            for ans in data['pos_answers']:
                if(len(ans)>3 and len(data['question'])>3):
                    self.questions_answer_pairs += [QAPair(data['id'], (data['question'],data['question_cui']), 
                                                            str(ans['a_id']), (ans['a_t'],ans['a_cui']),1)]
            for ans in data['neg_answers']:
                if(len(ans)>3 and len(data['question'])>3):
                    self.questions_answer_pairs += [QAPair(data['id'], (data['question'],data['question_cui']), 
                                                            str(ans['a_id']), (ans['a_t'],ans['a_cui']),0)]
        return self.questions_answer_pairs

    
class DataSetFactory():
    @staticmethod
    def loadDataSet(targetclass, **kwargs):
        return globals()[targetclass](**kwargs)