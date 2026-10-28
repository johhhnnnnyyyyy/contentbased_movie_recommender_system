import ast

import numpy as np
import pandas as pd


def _genre_labels(value):
    if isinstance(value, list):
        entries = value
    elif isinstance(value, str):
        try:
            entries = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    else:
        return []
    if not isinstance(entries, list):
        return []
    return [str(entry["name"]).strip() for entry in entries if isinstance(entry, dict) and entry.get("name")]


def build_catalog(movies):
    required = {"id", "title", "genres"}
    missing = required.difference(movies.columns)
    if missing:
        raise ValueError(f"Catalog is missing columns: {', '.join(sorted(missing))}")
    catalog = movies.loc[:, ["id", "title", "genres"]].copy()
    catalog = catalog.dropna(subset=["id", "title"])
    catalog["title"] = catalog["title"].astype(str).str.strip()
    catalog = catalog[catalog["title"].ne("")]
    catalog["catalog_id"] = catalog["id"]
    catalog["genre_labels"] = catalog["genres"].map(_genre_labels)
    return catalog.loc[:, ["catalog_id", "id", "title", "genre_labels"]].reset_index(drop=True)


def filter_catalog(catalog, query="", genre="All genres"):
    filtered = catalog.copy()
    normalized_query = str(query or "").strip()
    if normalized_query:
        filtered = filtered[filtered["title"].str.contains(normalized_query, case=False, na=False, regex=False)]
    if genre and genre != "All genres":
        normalized_genre = str(genre).casefold()
        filtered = filtered[
            filtered["genre_labels"].map(
                lambda labels: any(label.casefold() == normalized_genre for label in labels)
            )
        ]
    return filtered.sort_values("title", kind="stable").reset_index(drop=True)


def recommend(similarity, movies, movie_id, limit=5):
    ids = movies["id"].astype(str)
    matches = np.flatnonzero(ids.eq(str(movie_id)).to_numpy())
    if len(matches) == 0:
        raise ValueError("The selected movie is not included in the trained recommender.")
    selected_index = int(matches[0])
    scores = np.asarray(similarity[selected_index], dtype=float)
    ranked_indices = np.argsort(-scores, kind="stable")
    ranked_indices = [index for index in ranked_indices if index != selected_index][:limit]
    results = movies.iloc[ranked_indices].loc[:, ["id", "title"]].copy()
    results["similarity_score"] = [round(float(scores[index]), 3) for index in ranked_indices]
    return results.reset_index(drop=True)