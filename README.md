# Shri AI

One multi-task model, five domains (ML, DL, DS, NLP, CV), a general-purpose chatbot, and an admin panel for labeling data and retraining.

## Run

```bash
pip install -r requirements.txt
python app.py
```

- Chat: http://127.0.0.1:5000/
- Admin: http://127.0.0.1:5000/admin (password: `shri@ai`)

## Database

`DATABASE_URL` is read from `.env` (Neon serverless Postgres). Without it, a local SQLite file is used.

## First-time model

```bash
python -m src.train
```
