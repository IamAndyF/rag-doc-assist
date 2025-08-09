from langchain.retrievers import EnsembleRetriever

def get_ensemble_retriever(dense_retriever,sparse_retriever):
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4],
    )
    return ensemble_retriever
