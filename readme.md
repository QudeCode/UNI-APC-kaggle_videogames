# Video Game Score Prediction - Machine Learning Project

## Overview

This project focuses on predicting the critical scores of video games using advanced machine learning techniques. By analyzing various features such as global sales, user counts, and game-related metadata, the model provides insights into factors influencing the success of video games in the market. The goal is to build a robust predictive model that can help game developers and marketers anticipate a game’s reception before its release.

### Key Highlights

- **Techniques Used**: Data Preprocessing, Feature Selection, Regression Modeling, Hyperparameter Tuning
- **Modeling Approach**: Gradient Boosting Regressor (GBR) for high-performance prediction
- **Primary Objective**: Predict critic and user scores, which are crucial in determining a game's market performance.
- **Tools and Libraries**: Python, Scikit-learn, Pandas, Matplotlib, NumPy

## Table of Contents

- [Project Description](#project-description)
- [Techniques and Methodology](#techniques-and-methodology)
- [Experiments & Results](#experiments--results)
- [Data](#data)
- [Folder Structure](#folder-structure)
- [Installation and Setup](#installation-and-setup)
- [Results and Conclusion](#results-and-conclusion)
- [License](#license)

## Project Description

The video game industry is a competitive market where developers need to understand the potential reception of their games to make strategic decisions. This project aims to predict critical and user scores of video games based on features such as global sales, platform, publisher, and developer data. Using machine learning techniques, the model helps predict how well a game might perform based on historical data and known features.

## Techniques and Methodology

### Data Preprocessing
The first step involved cleaning and preparing the data:
- **Handling missing values**: Columns with missing data, such as critic and user scores, were cleaned by filling or removing missing entries.
- **Categorical Encoding**: Categorical features (e.g., Platform, Genre, Publisher) were encoded to numeric values using appropriate techniques to enable their use in machine learning models.
- **Feature Selection**: Based on correlation analysis, irrelevant or less impactful features (like 'Year_of_Release' and 'Genre') were removed.

### Model Selection & Evaluation
Multiple regression models were evaluated:
1. **Linear Regression**: Served as a baseline model.
2. **Decision Tree Regressor (DTR)**: Evaluated for its non-linear decision boundaries.
3. **Random Forest Regressor (RFR)**: Used for ensemble learning to improve accuracy.
4. **XGBoost Regressor (XGBR)**: Applied for gradient boosting and optimized for high accuracy.
5. **Gradient Boosting Regressor (GBR)**: Selected as the best-performing model with the highest validation scores.
6. **AdaBoost Regressor (ABR)**: Another ensemble method considered for comparison.

### Hyperparameter Tuning
The **GradientBoostingRegressor (GBR)** was selected for its superior performance, and its hyperparameters were optimized using **GridSearchCV**, ensuring the best possible configuration for making accurate predictions.

### Model Performance
The final model was evaluated using:
- **Mean Squared Error (MSE)**: To assess the accuracy of the predicted scores.
- **R-squared (R²)**: To determine how well the model explains the variance in the data.

### Final Model Results
- **MSE**: 1.311
- **R-squared**: 0.340

These results indicate that the GBR model is capable of making fairly accurate predictions, with room for further improvements.

## Experiments & Results

### Key Findings
- **Data Quality**: After handling missing values and encoding categorical variables, the dataset was reduced from 7905 rows to 5470, providing a clean foundation for model training.
- **Model Comparison**: The Gradient Boosting Regressor (GBR) outperformed other models such as Decision Trees, Random Forest, and XGBoost with a higher R-squared score and lower MSE.
- **Hyperparameter Tuning**: Optimizing the GBR model using GridSearchCV improved its performance significantly, validating the importance of hyperparameter optimization in machine learning.

### Results Visualization
- **Correlation Analysis**: Plots showing relationships between different features and critic/user scores.
- **Model Performance Metrics**: Detailed charts showing MSE and R² scores for different models.

## Data

The dataset used contains detailed information about video game sales, critical reception, and user feedback. The key features include:
- **Sales Data**: Global, North America, Europe, Japan, and other regions.
- **Game Information**: Developer, Publisher, Platform, Release Year, Genre.
- **Scores**: Critic scores and user scores.
  
### Data Sources:
- Kaggle's Video Games Sales dataset
- Various encoding files for developers, genres, and publishers

## Installation and Setup

### Prerequisites:
- Python 3.6+
- Jupyter Notebook (for notebook-based analysis)
- Required libraries:
  - Pandas
  - Scikit-learn
  - NumPy
  - Matplotlib
  - Seaborn

### Installation Steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo-name/video-game-score-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd video-game-score-prediction
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the analysis and scripts:
    - Start by running the exploratory data analysis (EDA) and preprocessing scripts.
    - Then move on to model selection and tuning.

## Results and Conclusion

The **Gradient Boosting Regressor (GBR)** model achieved the highest performance among the models tested, providing promising results with an R-squared score of 0.340 and an MSE of 1.311. The project illustrates the potential of machine learning in predicting video game success based on a variety of features. 

Future improvements could involve exploring additional features, experimenting with more complex models, or integrating more advanced techniques such as neural networks.

This project is a demonstration of practical machine learning application in the entertainment industry, showcasing skills in data preprocessing, feature engineering, model selection, and hyperparameter tuning.


