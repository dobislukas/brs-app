import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def get_k_nearest_neighbours_model(embeddings, metric="cosine"):
    knn_model = NearestNeighbors(metric=metric, n_jobs=-1)
    knn_model.fit(embeddings)
    return knn_model


def get_k_neighbours_for_vector(vector, knn_model, k=5):
    _, cos_indices = knn_model.kneighbors(
        vector, n_neighbors=k)
    return cos_indices
 

def get_k_recommendations_for_set_of_ids(
    set_of_ids,
    embeddings,
    k):

    # Prepare knn model
    knn_model = get_k_nearest_neighbours_model(embeddings, metric="cosine")

    # For each book ID query find recommended books IDs
    recommendation_dict = {}
    for book_emb_id in set_of_ids:

        book_embedding = embeddings[book_emb_id].reshape(1,-1)

        recommended_book_emb_ids = get_k_neighbours_for_vector(
            book_embedding, knn_model,
            k=k + 1)

        # Leave out first recommended ID as that is ID of queried book
        recommendation_dict[book_emb_id] = recommended_book_emb_ids[0,1:]

    return recommendation_dict

