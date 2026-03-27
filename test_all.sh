uv sync
source .venv/bin/activate
# Run unit tests
pytest
python lf_tests/lf-testing.py --model bert-tiny # Executes the lossy.fit() method on the IMDB-trained BERT-Tiny model
python lf_tests/lf-testing.py --model bert-base # Executes the lossy.fit() method on the MNLI-trained BERT-Base model
python lf_tests/lf-testing.py --model roberta # Executes the lossy.fit() method on the MNLI-trained RoBERTa-Base model

# Comment one of the two above if you only want to see one running!