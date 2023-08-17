#!usr/bin/python3

from typing import Annotated

import pickle
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

from utils import get_k_recommendations_for_set_of_ids


pickle_path = "data/"
embeddings = pd.read_pickle(pickle_path + "book_embeddings.pkl")
books = pd.read_pickle(pickle_path + "book_metadata.pkl")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
	return templates.TemplateResponse("index.html", 
		{"request": request}
	 )

@app.get("/list")
async def list_ui(request: Request):
	return templates.TemplateResponse(
	"list.html", 
	{"request": request, 
	 "table_df": books[["Book-Title", "Book-Author", 
	 					"Year-Of-Publication", "Publisher","ISBN"]].sort_values(
	 					by="Year-Of-Publication").to_html(
	 					classes="table table-bordered", index=False)})

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

