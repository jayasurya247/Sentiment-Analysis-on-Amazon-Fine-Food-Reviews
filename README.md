# Sentiment Analysis of Product Reviews

## Overview
This repository contains a sentiment analysis project that uses machine learning to classify product reviews as either positive or negative. The analysis is performed on a dataset containing reviews from Amazon customers. Various machine learning models including Logistic Regression, Naive Bayes, Decision Trees, and Random Forests are evaluated to determine which performs best at predicting sentiment based on text content.

## Repository Structure
- `Reviews.csv`: The dataset file containing review texts and scores. To download the dataset go the [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
- `sentiment_analysis.ipynb`: Jupyter notebook containing all the analyses and visualizations.

## Installation
To run this project, you need Python 3.x and the following libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- nltk


## Data
The dataset includes the following key columns:
- `Text`: The text of the review.
- `Score`: The rating given by the user, used to derive the sentiment.

## Features
- Text preprocessing includes cleaning, tokenization, and lemmatization.
- Sentiment scores derived using NLTK's VADER.
- Tfidf vectorization for converting text to numeric form suitable for ML model training.

## Usage
To run the sentiment analysis, open the Jupyter notebook `sentiment_analysis.ipynb` and execute the cells sequentially.

## Models Implemented
- Logistic Regression
- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- Decision Tree
- Random Forest

## Results
The effectiveness of each model is evaluated based on accuracy, precision, recall, and F1-score. Results are visually presented through a bar chart comparing the performance of the models.

## Contributing
Contributions to this project are welcome! You can contribute in the following ways:
- Enhancing the preprocessing pipeline.
- Experimenting with different models or tuning the hyperparameters.
- Improving the visualization of results.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## References
- [Amazon Fine Food Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- [Python Lemmatization with NLTK - GeeksforGeeks](https://www.geeksforgeeks.org/python-lemmatization-with-nltk/)
- [NLTK Sentiment VADER - NLTK Documentation](https://www.nltk.org/_modules/nltk/sentiment/vader.html)
- [Access to numbers in classification report - Stack Overflow](https://stackoverflow.com/questions/48417867/access-to-numbers-in-classification-report-sklearn)

## Contact
- [Jaya Surya Thota](https://github.com/jayasurya247) - feel free to contact me!


