# -------- File: retrievers/__init__.py --------
from .base import BaseRetriever
from .bm25_retriever import BM25Retriever
from .dot_retriever import DotProductRetriever
from .faiss_retriever import FaissRetriever
from .faiss_lsh_retriever import FaissLSHRetriever
from .annoy_lsh_retriever import AnnoyLSHRetriever
from .nmslib_lsh_retriever import NMSLIBLSHRetriever
# from .scann_retriever import ScaNNRetriever
RETRIEVER_CLASSES = {
    "bm25": BM25Retriever,
    "dot": DotProductRetriever,
    "faiss": FaissRetriever,
    "lsh_faiss": FaissLSHRetriever,
    "lsh_annoy": AnnoyLSHRetriever,
    "lsh_nmslib": NMSLIBLSHRetriever,
    # "scann": ScaNNRetriever,
}
