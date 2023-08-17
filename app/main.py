#!usr/bin/python3
import pickle
import numpy as np
import pandas as pd

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from typing import Annotated

from sklearn.neighbors import NearestNeighbors

pickle_path = "data/"
embeddings = pd.read_pickle(pickle_path + "book_embeddings.pkl")
books = pd.read_pickle(pickle_path + "book_metadata.pkl")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

def convert_temp_ids_to_book_ids(ratings, temp_ids):
    recommended_book_ids = ratings[
        ratings["Book-Embedding-ID"].isin(temp_ids[0])]

    sorted_recommended_book_ids = recommended_book_ids.sort_values(
        by=["Book-Embedding-ID"],
        key=lambda x: x.map(
            {v: i for i, v in enumerate(temp_ids[0])}))

    sorted_recommended_book_ids = sorted_recommended_book_ids["Book-ID"].unique()
    return sorted_recommended_book_ids

def get_book_titles_from_book_ids(books_metadata, book_ids):

    recommended_books = books_metadata[
        books_metadata['Book-ID'].isin(book_ids)].sort_values(
            by=["Book-ID"],
            key=lambda x: x.map({v: i for i, v in enumerate(book_ids)}))

    return recommended_books['Book-Title'].unique()

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

@app.get("/")
async def index(request: Request):
	return templates.TemplateResponse("index.html", 
		{"request": request}
	 )

@app.get("/list")
async def list_ui(request: Request):
	return templates.TemplateResponse("list.html", {"request": request, "table_df": books[["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher","ISBN"]].sort_values(by="Year-Of-Publication").to_html(classes="table table-bordered", index=False)})

@app.get("/contact")
async def contact_ui(request: Request):
	return templates.TemplateResponse("contact.html", {"request": request})

@app.post("/recommend_books")
async def recommend(request: Request, user_input: Annotated[str, Form()]):
	
	### START This part should be changed to utilize Weaviate database
	book_name = user_input
	book_id = books[books["Book-Title"] == book_name]["Book-Embedding-ID"].values[0]
	pca_recommend_dict = get_k_recommendations_for_set_of_ids(
		set_of_ids=[book_id],
		embeddings=embeddings,
		k=12)

	recommendations = list(pca_recommend_dict.values())[0]
	### END
	
	### START This part should be supplemented by MySQL/PostgreSQL database
	data = []
	for similar_book_id in recommendations:
		item = []
		
		similar_book = books[books["Book-Embedding-ID"] == similar_book_id]
		
		item.extend(list(similar_book["Book-Title"].values))
		item.extend(list(similar_book["Book-Author"].values))
		item.extend(list(similar_book["Image-URL-L"].values))
		data.append(item)
	### END
	
	return templates.TemplateResponse("index.html", 
		{"request": request,
		 "data": data}
	)

