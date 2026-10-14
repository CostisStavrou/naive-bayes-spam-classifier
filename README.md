# Naive Bayes Spam Classifier

A Bernoulli naive Bayes classifier written from scratch in NumPy. It decides whether an email is spam from 54 binary keyword and character features.

**Built:** May 2024, as Year 1 coursework for my Computer Science degree at the University of Bath.
**Revisited:** October 2026. I fixed a bug in the prediction step, vectorised the classifier, and added an evaluation against baselines (details below).

## Results

Accuracy on a held-out test set of 500 emails:

| Model | Test accuracy |
|---|---|
| Always predict "not spam" (majority class) | 60.2% |
| **This classifier (from scratch)** | **88.8%** |
| scikit-learn `BernoulliNB` (same model, as a check) | 88.8% |
| scikit-learn `LogisticRegression` (reference) | 92.6% |

Precision is 91.3% and recall is 79.4%. When it flags an email as spam it is usually right, but it misses about one spam email in five.

## How it works

- **Training** estimates the prior probability of spam and ham, and for each feature the probability that it appears in each class. Laplace (add-one) smoothing stops unseen features from producing zero probabilities.
- **Prediction** sums log probabilities for each class and picks the larger. Working in log space avoids numerical underflow from multiplying 54 small numbers.
- Naive Bayes assumes the features are independent given the class. That is why logistic regression, which makes no such assumption, scores higher: words like *free* and *money* tend to appear together, so naive Bayes counts that evidence twice.

## What changed in 2026

The original `predict` method skipped feature 0 because it assumed column 0 was still the label. Every other feature was then scored with its neighbour's probabilities. Fixing the alignment raised accuracy from **83.2% to 88.8%**, matching scikit-learn. The commit history keeps the original version.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook spam_classifier.ipynb
```

## Data

The `data/` folder holds 1,000 training and 500 test emails. Each row is a label followed by 54 binary features. The coursework provided the data, a binarised subset of the [UCI Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) (CC BY 4.0).
