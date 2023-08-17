# Book Recommendation System

Recommend similar books based on your query in form of book title.

Currently (temporarily) hosted on: https://brs-app.onrender.com/

Number of supported book titles is limited but can be browsed in "List of books" part of webpage.

## Overview

* notebooks/ - Series of jupyter notebooks detailing creation process of
            book embeddings from Book-Crossing Dataset.
* app/ - FastAPI application offering Nearest Neighbor recommendations based on
         created book embeddings.
* pres/ - Summary of book recommendation system solution, offers short overview
          of aforementioned notebooks and then deals with productionalization
          of upscaled arbitrary book recommendation solution.

## How to run

### Local - without docker

1.  Install packages from requirements.txt 
2.  Run following code inside app directory

```
sudo uvicorn main:app --host 0.0.0.0 --port 80 --reload
```
3. Webpage is now available on your device at http://0.0.0.0:80

### Local - with docker
1. Inside project directory build image container.
```
docker build . -t brs-image
```
2. Inside project directory build image container.
```
docker run -p 80:80 brs-image
```
3. Webpage is now available on your device at http://0.0.0.0:80

### Links
Webpage: https://brs-app.onrender.com/

Book-Crossing Dataset: http://www2.informatik.uni-freiburg.de/~cziegler/BX/

