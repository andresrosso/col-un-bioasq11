from haystack.document_stores import BaseKnowledgeGraph
from haystack.document_stores import KeywordDocumentStore
from haystack.nodes.retriever.base import BaseGraphRetriever
from haystack.nodes.retriever.base import BaseRetriever
from typing import List, Dict, Union, Optional, Any
from haystack.schema import Document, FilterType
from haystack.document_stores import BaseDocumentStore
from elasticsearch import Elasticsearch
import sys

sys.path.append('../')
import globals
from elastic_search_utils import elastic_utils
import bioasq_eval

class BioASQ_Retriever(BaseRetriever):
    def __init__(
        self,
        top_k: int = 10,
        document_store: Optional[BaseDocumentStore] = None
    ):
        super().__init__()
        self.document_store = document_store
        self.top_k = top_k
        self.es = Elasticsearch(globals.ES.server)
    
    def es_hist_to_document(
        es_hits
    ):
        doc_list = []
        for hit in es_hits:
            score = hit['_score']
            id = hit['_source']['pmid']
            content = hit['_source']['abstract']
            meta = hit['_source']
            doc = Document(id=id, content=content, meta=meta)
            doc_list.append(doc)
        return doc_list
            
        
    def retrieve(
            self,
            query: str,
            filters: Optional[FilterType] = None,
            top_k: Optional[int] = None,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            scale_score: Optional[bool] = None,
            document_store: Optional[BaseDocumentStore] = None,
        ) -> List[Document]:
            document_store = document_store or self.document_store
            if document_store is None:
                raise ValueError(
                    "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
                )
            if not isinstance(document_store, KeywordDocumentStore):
                raise ValueError("document_store must be a subclass of KeywordDocumentStore.")
            if top_k is None:
                top_k = self.top_k
            res = elastic_utils.search_doc_by_query(
                        question = query,
                        fields = globals.ES.search_fields,
                        size = top_k,
                        es_client = self.es,
                        index = globals.ES.index
                    ) 
            docs = BioASQ_Retriever.es_hist_to_document(res['hits']['hits'])
            return docs

    def retrieve_batch(
            self,
            queries: List[str],
            filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
            top_k: Optional[int] = None,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            batch_size: Optional[int] = None,
            scale_score: Optional[bool] = None,
            document_store: Optional[BaseDocumentStore] = None,
        ) -> List[List[Document]]:
        return [['a','b']]