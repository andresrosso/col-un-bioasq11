import os

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Batch, Data

from src.elastic_search_utils.elastic_utils import load_json

class GraphDataset:
    METADATA_FILE = 'metadata.parquet'
    DEBUG_SIZE = 128
    
    def __init__(
        self,
        dataset_path,
        batch_size,
        val_percentage,
        test_percentage,
        random_state,
        score_threshold=None,
        debug=False
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage
        self.random_state = random_state
        self.score_threshold = score_threshold

        self.random_generator = np.random.RandomState(
            random_state
        )
        
        self.metadata = pd.read_parquet(
            f'{dataset_path}/{self.METADATA_FILE}'
        )
        
        if debug:
            self.metadata = self.metadata[
                self.metadata['question_id'].isin(
                    self.metadata['question_id'].unique()[:self.DEBUG_SIZE]
                )
            ]
        self.__add_document_path()
        
        if score_threshold is not None:
            self.__relabel()
        
        self.splits = self.__train_val_test_split()
        
    def __add_document_path(self):
        self.metadata['path'] = self.metadata.apply(
            lambda document: \
                f"{self.dataset_path}/{document['question_id']}_{document['document_id']}.json",
            axis=1
        )
    
    @staticmethod
    def __relabel_document(document, score_threshold):
        if (
            (document['origin'] == 'original') or
            (document['score'] >= score_threshold)
        ):
            return 1.0

        return 0.0
        
    def __relabel(self):
        self.metadata['label'] = self.metadata.apply(
            lambda document: self.__relabel_document(
                document, self.score_threshold
            ),
            axis=1
        )
        
    def __train_val_test_split(
        self
    ):
        unique_question_ids = \
            self.metadata['question_id'].unique()
        train_questions, val_questions, test_questions = \
            self.__train_val_test_split_questions(
                unique_question_ids
            )
        return {
            "train": self.metadata[
                self.metadata['question_id'].isin(
                    train_questions
                )
            ].copy(deep=True),
            "val": self.metadata[
                self.metadata['question_id'].isin(
                    val_questions
                )
            ].copy(deep=True),
            "test": self.metadata[
                self.metadata['question_id'].isin(
                    test_questions
                )
            ].copy(deep=True)
        }
    def __train_val_test_split_questions(
        self,
        questions
    ):  
        shuffled_questions = np.copy(questions)
        self.random_generator.shuffle(shuffled_questions)
        train_percentage = 1.0 - (
            self.val_percentage + self.test_percentage
        )
        test_split = 1.0 - self.test_percentage

        train, val, test = np.split(
            shuffled_questions,
            [
                int(train_percentage * len(questions)),
                int(test_split * len(questions))
            ]
        )
        return train, val, test
    
    def __get_example_graph(self, path):
        raw_graph = load_json(path)
        raw_graph['label'] = self.__relabel_document(
            raw_graph, self.score_threshold
        )
        x = torch.tensor(raw_graph['similarity_matrix'], dtype = torch.float)
        y = torch.tensor(raw_graph['label'], dtype = torch.long)
        edge_index = torch.tensor(raw_graph['edges'],dtype = torch.long)
        graph = Data(x=x, edge_index=edge_index, y=y)
        return graph

    def __get_batch_graphs(self, batch_paths):
        graphs = []
        for path in batch_paths:
            example_graph = self.__get_example_graph(
                path
            )
            graphs.append(example_graph)
            
        
        return Batch.from_data_list(graphs)

    def get_batch(self, split_type=None):
        
        if split_type is not None:
            df_data = self.splits[split_type].copy(deep=True)
        else:
            df_data = self.metadata.copy(deep=True)

        if split_type == 'train':
            df_data = df_data.sample(
                frac=1,
                random_state=self.random_state
            )
        
        n_splits = np.ceil(len(df_data)/self.batch_size)
        batches = np.array_split(df_data, n_splits)

        for batch in batches:
            batch_paths = batch['path'].tolist()
            right_train_shape = (
                (split_type == 'train') and
                (len(batch_paths) == self.batch_size)
            )
            predict_all = split_type is None
            is_train_val_split = split_type in ['val', 'test']
            if right_train_shape or predict_all or is_train_val_split:
                batch_graphs = self.__get_batch_graphs(
                    batch_paths
                )

                yield batch_graphs
