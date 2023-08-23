#!usr/bin/python3

# Imports
import pandas as pd

from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates

from utils import get_k_recommendations_for_set_of_ids

#TODO This should be read from config
pickle_path = "data/"
embeddings = pd.read_pickle(pickle_path + "book_embeddings.pkl")
books = pd.read_pickle(pickle_path + "book_metadata.pkl")
number_of_recommendations_for_web = 12

# FastAPI initialization
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_model=None)
async def index(request: Request) -> templates.TemplateResponse:
    """Renders the index.html webpage."""
    return templates.TemplateResponse(
        "index.html", {"request": request})


@app.get("/list", response_model=None)
async def list_ui(request: Request) -> templates.TemplateResponse:
    """
    Renders the list.html webpage,
    with table of supported book titles
    for query.
    """

    return templates.TemplateResponse(
        "list.html", {"request": request,
                      "table_df": books[
                        ["Book-Title", "Book-Author",
                         "Year-Of-Publication", "Publisher",
                         "ISBN"]].sort_values(
                             by="Year-Of-Publication").to_html(
                             classes="table table-bordered", index=False)})


@app.get("/contact", response_model=None)
async def contact_ui(request: Request) -> templates.TemplateResponse:
    """Renders the contact.html webpage."""
    return templates.TemplateResponse("contact.html", {"request": request})


@app.get("/recommend_books", response_model=None)
async def recommend(
        request: Request,
        user_input: str = Query(...)) -> templates.TemplateResponse:
    """
    Return recommended books based on the given query book title.

    :param Request: contains book embeddings
    :param user_input: input book title for recommendation
    :return: recommended books with their information
    """

    # START This part should be changed to utilize Weaviate database
    book_name = user_input
    book_id = books[
        books["Book-Title"] == book_name]["Book-Embedding-ID"].values[0]

    pca_recommend_dict = get_k_recommendations_for_set_of_ids(
        set_of_ids=[book_id],
        embeddings=embeddings,
        k=number_of_recommendations_for_web)

    recommendations = list(pca_recommend_dict.values())[0]
    # END

    # START This part should be supplemented by MySQL/PostgreSQL database
    data = []
    for similar_book_id in recommendations:

        item = []
        similar_book = books[books["Book-Embedding-ID"] == similar_book_id]

        item.extend(list(similar_book["Book-Title"].values))
        item.extend(list(similar_book["Book-Author"].values))
        item.extend(list(similar_book["Image-URL-L"].values))

        data.append(item)
    # END

    return templates.TemplateResponse(
        "index.html", {"request": request, "data": data}
    )
