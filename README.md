# TransferValuePredictor

transfer-value-predictor/
├── data/
│   ├── raw/              # untouched scraped/pulled data, per season
│   └── processed/        # cleaned, merged dataset
├── src/
│   ├── scrape_transfermarkt.py   # market values per player per season
│   ├── fetch_fbref_stats.py      # goals, assists, minutes, position, age
│   ├── merge_data.py             # join sources, fuzzy name-match
│   ├── train_model.py            # linear regression + eval
│   └── predict.py                # "what are you worth" fun tool
├── notebooks/
│   └── exploration.ipynb         # optional, for EDA/plots during dev
├── requirements.txt
└── README.md
