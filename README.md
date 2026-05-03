# MLOps Full Model

Multi-task MLOps demo project controlled by `MODEL_TYPE`.

Supported values:

- `classification`
- `regression`
- `clustering`
- `time_series`

## Project Structure

```text
mlops-full-model/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── train.py
│   └── tasks/
├── .env
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Change Model Type

Edit `.env`:

```env
MODEL_TYPE=classification
```

or set the environment variable in PowerShell:

```powershell
$env:MODEL_TYPE="classification"
```

## Run Locally

```powershell
python src/data_preprocessing.py
python src/train.py
uvicorn api.main:app --reload
```

In another terminal:

```powershell
streamlit run app/streamlit_app.py
```

## Run With Docker Compose

```powershell
$env:MODEL_TYPE="classification"
docker compose up --build
```

The `trainer` service preprocesses data and trains the selected model before the API and UI start.

## MLflow UI

After training:

```powershell
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```
