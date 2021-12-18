# TMDb movie data

*cleaned from original data on [Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata)*

This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue.

- Certain columns, like `cast` and `genres`, contain multiple values separated by pipe (|) characters.
- There are some odd characters in the `cast` column. We can leave them as is.
- The final two columns ending with `_adj` show the budget and revenue of the associated movie in terms of 2010 dollars, accounting for inflation over time.

**Some questions to explore:**
- Which genres are most popular from year to year? 
- What kinds of properties are associated with movies that have high revenues?