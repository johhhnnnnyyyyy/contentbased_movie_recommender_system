# ReelMatch Movie Recommendation System

ReelMatch is a Streamlit app that recommends movies using content similarity. Select a film from the catalogue and the app returns movies with similar content signals, including genres and other information used by the trained recommender.

## Live demo

The app is designed to run on [Streamlit Community Cloud](https://streamlit.io/cloud). After deployment, Streamlit provides a URL in the format:

```text
https://<your-app-name>.streamlit.app
```

Add the deployed URL here once it is available:

**Live demo:** Not deployed yet

### Deploy your own live demo

1. Make sure the project is pushed to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Select the repository:
   `johhhnnnnyyyyy/contentbased_movie_recommender_system`
4. Select the `main` branch.
5. Set the main file to `app.py`.
6. Click **Deploy**.

The trained model is intentionally excluded from Git because it is approximately 7 GB. Streamlit Cloud will not be able to start the recommendation feature until `movie_recommender.joblib` is made available to the deployed app. Store that file in external object storage or another artifact service, then download it during deployment/startup. Do not commit credentials or private download URLs to the repository.

## Run locally

### Requirements

- Python 3.10 or newer
- Git
- The project dataset files
- The trained `movie_recommender.joblib` model file

### 1. Clone the repository

```powershell
git clone https://github.com/johhhnnnnyyyyy/contentbased_movie_recommender_system.git
cd contentbased_movie_recommender_system
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation for the current terminal, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy joblib
```

### 4. Add the trained model

Place the trained model at the project root:

```text
contentbased_movie_recommender_system/
├── app.py
├── recommender_service.py
├── movie_recommender.joblib
└── dataset/
    └── movies_metadata.csv
```

The app expects `movie_recommender.joblib` to contain a dictionary with:

- `similarity`: the saved similarity matrix
- `movies`: a pandas DataFrame containing `id` and `title` columns

The model is not included in this repository because of its size. The full source dataset is available on [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

### 5. Start the app

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, normally:

```text
http://localhost:8501
```

Open that URL in your browser, search for a movie, select the exact title, and click **Build my watchlist**.

## Project structure

```text
app.py                                Streamlit user interface
recommender_service.py                Catalogue filtering and recommendations
CONTENT-movie_recommender_system.ipynb  Model/data preparation notebook
dataset/movies_metadata.csv           Movie catalogue used by the app
dataset/keywords.csv                 Training dataset retained in the repository
dataset/links.csv                    Training dataset retained in the repository
movies.md                             Dataset source note
.streamlit/config.toml               Streamlit theme configuration
```

Large local files are excluded by [`.gitignore`](.gitignore), including the trained model, ratings data, credits data, virtual environments, caches, and Python bytecode.

## Troubleshooting

### `movie_recommender.joblib was not found`

Copy the model into the project root and confirm the filename is exactly:

```text
movie_recommender.joblib
```

### `dataset/movies_metadata.csv was not found`

Confirm that the file exists at:

```text
dataset/movies_metadata.csv
```

### The app does not open

Make sure the virtual environment is activated and run:

```powershell
python -m streamlit run app.py
```

If Streamlit is not installed, install the dependencies again:

```powershell
python -m pip install streamlit pandas numpy joblib
```

## Data and attribution

The movie data comes from [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Review Kaggle's dataset terms and the original data licenses before redistributing or deploying the data.
