# text_clf
Text classification

# requirements
Python 3.8.5

```
nltk==3.5
numpy==1.18.5
tensorflow==2.3.0
beautifulsoup4==4.9.3
```

# how to run
`main.py --mode train --model_path models/best_model_lstm_5k`

`main.py --mode test --model_path models/best_model_lstm_5k`

`main.py --mode label --model_path models/best_model_lstm_5k --input_text_file data/text_doc_grain_rice.txt`
