
```
APC_kaggle_videogames
├─ data
│  ├─ data_table.txt
│  ├─ encodings
│  │  ├─ Developer_encoding.csv
│  │  ├─ Genre_encoding.csv
│  │  ├─ Platform_encoding.csv
│  │  └─ Publisher_encoding.csv
│  ├─ test.csv
│  ├─ train.csv
│  ├─ train_DEF.csv
│  ├─ train_tracted.csv
│  └─ Video_Games_Sales_as_at_22_Dec_2016.csv
├─ encodings
│  ├─ Developer_encoding.csv
│  ├─ Genre_encoding.csv
│  ├─ Platform_encoding.csv
│  └─ Publisher_encoding.csv
├─ informe
│  ├─ 1600215_kaggle_informe.docx
│  ├─ figs
│  ├─ ~$00215_kaggle_informe.docx
│  └─ ~WRL0003.tmp
├─ readme.md
├─ results
│  ├─ 1_EDA
│  │  ├─ 10_dataset_analysis.txt
│  │  ├─ 10_first_correlation_matrix.png
│  │  ├─ 11_critics_balance_analysis.png
│  │  ├─ 12_user_count_balance_high.png
│  │  └─ 12_user_count_balance_low.png
│  ├─ 2_Preprocessing
│  │  ├─ 20_initialization.txt
│  │  ├─ 21_NaNs_and_Encoding.txt
│  │  ├─ 22_correlations_after_preprocessing_full.png
│  │  └─ correlation_matrix_after_preprocessing.csv
│  ├─ 3_Metric_selection
│  │  ├─ 30_filter_data.txt
│  │  └─ 31_last_correlation.png
│  └─ 4_ModelSelection_and_Crossvalidation
│     ├─ 40_linear_regressions.txt
│     ├─ 40_linear_regression_plot.png
│     ├─ 41_ABR_regression_results.png
│     ├─ 41_DTR_regression_results.png
│     ├─ 41_GBR_regression_results.png
│     ├─ 41_LR_regression_results.png
│     ├─ 41_regression_results.txt
│     ├─ 41_RFR_regression_results.png
│     ├─ 41_XGBR_regression_results.png
│     ├─ 42_GBR_results.txt
│     ├─ 44_GBR_predictions_train.png
│     ├─ 44_GBR_predictions_train_critic.png
│     └─ 45_GBR_predictions_train_user.png
└─ scripts
   ├─ 1_EDA
   │  ├─ 10_dataset_analysis.py
   │  ├─ 11_critics_analysis.py
   │  └─ 12_user_count_balance.py
   ├─ 2_Preprocessing
   │  ├─ 20_prepare_datasets.py
   │  ├─ 21_NaNs_and_Encoding.py
   │  ├─ 22_Normalize_scores_to_10.py
   │  └─ 23_correlations_after_preprocessing.py
   ├─ 3_Metric_selection
   │  ├─ 30_filter_data.py
   │  └─ 31_last_correlation.py
   ├─ 4_ModelSelection_and_Crossvalidation
   │  ├─ 40_linear_regression.py
   │  ├─ 41_regression_evaluation.py
   │  ├─ 42_GBR_adjustments.py
   │  └─ 43_GBR_regression.py
   └─ 5_Final_analysis
      └─ 50_final_analysis.py

```