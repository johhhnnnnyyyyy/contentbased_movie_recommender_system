from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from recommender_service import build_catalog, filter_catalog, recommend


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "movies_metadata.csv"
MODEL_PATH = ROOT / "movie_recommender.joblib"


def validate_model_bundle(bundle):
    if not isinstance(bundle, dict):
        raise ValueError("The saved model must be a joblib dictionary.")
    for key in ("similarity", "movies"):
        if key not in bundle:
            raise ValueError(f"The saved model is missing '{key}'.")
    movies = bundle["movies"]
    if not isinstance(movies, pd.DataFrame) or not {"id", "title"}.issubset(movies.columns):
        raise ValueError("The saved movie table must include id and title columns.")
    if getattr(bundle["similarity"], "shape", (0, 0))[0] != len(movies):
        raise ValueError("The similarity matrix does not match the saved movie table.")


@st.cache_data(show_spinner=False)
def load_catalog():
    if not DATASET_PATH.exists():
        raise FileNotFoundError("dataset/movies_metadata.csv was not found.")
    raw_movies = pd.read_csv(DATASET_PATH, usecols=["id", "title", "genres"], low_memory=False)
    return build_catalog(raw_movies)


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("movie_recommender.joblib was not found.")
    bundle = joblib.load(MODEL_PATH)
    validate_model_bundle(bundle)
    return bundle


def movie_label(row):
    genre_text = ", ".join(row.genre_labels[:2]) if row.genre_labels else "Genre unavailable"
    return f"{row.title}  |  {genre_text}  |  ID {row.catalog_id}"


def genre_text_for_movie(catalog, movie_id):
    matches = catalog[catalog["id"].astype(str).eq(str(movie_id))]
    if matches.empty:
        return "Genre unavailable"
    return ", ".join(matches.iloc[0].genre_labels) or "Genre unavailable"


def render_recommendations(results, catalog):
    for position, row in results.reset_index(drop=True).iterrows():
        genres = genre_text_for_movie(catalog, row.id)
        with st.container(border=True):
            rank_col, title_col, score_col = st.columns([0.45, 4, 1], vertical_alignment="center")
            rank_col.markdown(f"### {position + 1:02d}")
            title_col.subheader(row.title, divider=False)
            title_col.caption(genres)
            score_col.metric("Match", f"{row.similarity_score:.3f}")


def main():
    st.set_page_config(page_title="ReelMatch", page_icon=":material/movie:", layout="wide")
    st.html(
        """
        <style>
        .stApp { background: radial-gradient(circle at 84% 5%, rgba(240,100,78,.16), transparent 23rem), radial-gradient(circle at 12% 34%, rgba(244,181,94,.09), transparent 28rem), #0A0B10; }
        .stApp::before { content: ""; position: fixed; inset: 0; pointer-events: none; background: linear-gradient(115deg, rgba(255,255,255,.015), transparent 22%, rgba(255,255,255,.01) 70%, transparent); }
        .block-container { max-width: 1160px; }
        h1 { letter-spacing: -.045em; font-size: clamp(3.2rem, 7vw, 6.6rem) !important; line-height: .92; }
        h2, h3 { letter-spacing: -.025em; }
        [data-testid="stMetric"] { background: rgba(255,255,255,.035); border-radius: .8rem; padding: .8rem; }
        [data-testid="stVerticalBlockBorderWrapper"] { background: rgba(21,23,33,.76); backdrop-filter: blur(18px); }
        .stButton > button { min-height: 3.25rem; font-weight: 800; letter-spacing: .01em; }
        [data-testid="stTextInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div { min-height: 3.25rem; }
        </style>
        """
    )

    try:
        catalog = load_catalog()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Catalog unavailable: {error}")
        st.stop()

    st.session_state.setdefault("recommendation_results", None)
    st.session_state.setdefault("recommendation_title", None)

    hero_col, fact_col = st.columns([3.2, 1], vertical_alignment="bottom")
    with hero_col:
        st.caption(":material/local_movies: CONTENT-BASED DISCOVERY")
        st.title("Your next great watch")
        st.markdown("Choose one film. ReelMatch traces its story, themes, cast, and creative signals to uncover a more considered next pick.")
    with fact_col:
        with st.container(border=True):
            st.metric("Films indexed", f"{len(catalog):,}")
            st.caption("Precomputed content similarity")

    st.divider()
    st.subheader("Start with a film", divider=False)
    st.caption("Search the catalogue, set a genre if you want to narrow the field, then choose an exact title.")

    genres = sorted({genre for labels in catalog.genre_labels for genre in labels})
    with st.container(border=True):
        search_col, genre_col = st.columns([2.2, 1], vertical_alignment="bottom")
        query = search_col.text_input("Search the catalogue", placeholder="Try Jumanji, Interstellar, or Parasite", key="movie_search")
        genre = genre_col.selectbox("Genre", ["All genres", *genres], key="genre_filter")
    filtered = filter_catalog(catalog, query, genre)

    if not query.strip():
        st.info("Enter a title to open the curated match list.", title="Your film search starts here")
        return
    if filtered.empty:
        st.warning("Try a broader title or clear the genre filter to reveal more choices.", title="No matching films")
        return

    choices = filtered.head(100)
    with st.container(border=True):
        st.caption(f"{len(filtered):,} possible match{'es' if len(filtered) != 1 else ''}")
        selected_label = st.selectbox(
            "Choose the exact film",
            choices.apply(movie_label, axis=1).tolist(),
            key="movie_choice",
        )
        selected_movie = choices.loc[choices.apply(movie_label, axis=1).eq(selected_label)].iloc[0]
        request_recommendations = st.button(
            "Build my watchlist",
            icon=":material/auto_awesome:",
            type="primary",
            width="stretch",
        )

    if request_recommendations:
        try:
            with st.status("Opening the saved recommendation model...", expanded=False) as status:
                bundle = load_model_bundle()
                results = recommend(bundle["similarity"], bundle["movies"], selected_movie.catalog_id)
                status.update(label="Your cinematic matches are ready.", state="complete")
        except (FileNotFoundError, ValueError, MemoryError, OSError) as error:
            st.error(f"{error} The app reuses the saved model and never rebuilds the vectorizer.", title="Recommendations are unavailable")
            return
        st.session_state.recommendation_results = results
        st.session_state.recommendation_title = selected_movie.title

    if st.session_state.recommendation_results is not None:
        st.divider()
        st.caption(":material/featured_play_list: CURATED FOR YOU")
        st.header(f"Because you chose {st.session_state.recommendation_title}")
        st.caption("Match scores describe content similarity in the saved model. They are not audience ratings or review scores.")
        render_recommendations(st.session_state.recommendation_results, catalog)


if __name__ == "__main__":
    main()
