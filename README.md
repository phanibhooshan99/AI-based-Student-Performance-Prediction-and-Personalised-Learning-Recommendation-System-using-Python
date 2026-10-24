# 🎓 AI-Based Student Performance Prediction & Personalised Learning Recommendation System

A Python machine-learning project that **predicts a student's final score and performance level**, then generates a **personalised learning plan** (weak topics, curated resources, habit tips and a weekly study schedule).

## Features

- **Score prediction** (regression) and **performance level** (At-Risk / Average / High) classification
- Compares Linear/Logistic Regression, Random Forest and Gradient Boosting; saves the best models automatically
- **Personalised recommender**: detects weak topics, matches resource difficulty to mastery, gives habit tips and a weekly plan
- **Streamlit web app** and **CLI demo**
- Synthetic dataset generator (swap in your own CSV with the same columns)
- Unit tests with `pytest`

## Project structure

```
student-performance-ai/
├── app.py                  # Streamlit web app
├── main.py                 # CLI (generate / train / demo)
├── requirements.txt
├── data/
│   └── resources.csv       # learning-resource catalogue (students.csv is generated)
├── src/
│   ├── config.py           # paths, features, thresholds
│   ├── data_generator.py   # synthetic data
│   ├── train.py            # training + evaluation
│   ├── predict.py          # prediction + recommendation pipeline
│   └── recommender.py      # personalisation engine
├── tests/test_pipeline.py
├── models/                 # saved models (created by training)
└── reports/                # metrics.json + feature_importance.png
```

## Quick start

```bash
git clone https://github.com/<your-username>/student-performance-ai.git
cd student-performance-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py generate         # create data/students.csv
python main.py train            # train models, writes models/ and reports/
python main.py demo             # sample prediction + recommendations
streamlit run app.py            # launch the web app
pytest                          # run tests
```

## Using your own data

Create `data/students.csv` with these columns: `gender, study_hours, attendance, previous_score, assignment_score, quiz_score, participation, sleep_hours, internet_access, extra_activities, parental_support, topic_mathematics, topic_programming, topic_data_structures, topic_databases, topic_statistics, final_score, performance_level`, then run `python main.py train`.

## How it works

1. **Preprocessing** - numeric features are standardised, categorical features one-hot encoded (scikit-learn `Pipeline`).
2. **Modelling** - regression predicts `final_score`; classification predicts the level. Best model is chosen by R² / macro-F1.
3. **Recommendation** - topics scoring below 60 are flagged; difficulty is `beginner (<40)`, `intermediate (<65)` or `advanced`; resources come from `data/resources.csv`.

## Note

The bundled dataset is **synthetic** for demonstration, so reported metrics reflect the generator, not real students. Use real, consented and anonymised data for production use.

## License

MIT
