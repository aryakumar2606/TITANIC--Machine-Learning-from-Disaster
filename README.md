🚢 Titanic Survival Prediction — A Beginner's Guide to Machine Learning

Can we predict who survived the Titanic disaster using passenger data and machine learning? Yes — and this project shows you exactly how, step by step.


📖 Table of Contents

What Is This Project?
Why This Project Is Perfect for Beginners
The Dataset — What Are We Working With?
The Machine Learning Pipeline — Big Picture
Setup & Requirements
Step-by-Step Code Walkthrough

Step 1 — Importing Libraries
Step 2 — Loading the Data
Step 3 — Exploring the Data (EDA)
Step 4 — Handling Missing Values
Step 5 — Dropping Irrelevant Columns
Step 6 — Encoding Categorical Variables
Step 7 — Splitting Features and Target
Step 8 — Train-Validation Split
Step 9 — Training the Random Forest Model
Step 10 — Evaluating the Model
Step 11 — Making Predictions on Test Data
Step 12 — Generating the Submission File
Step 13 — Visualizations
Step 14 — Feature Importance


Results & Output
Key Concepts Explained Simply
Common Errors & Fixes
How to Improve This Further


🧠 What Is This Project?
This project solves a binary classification problem — for every passenger aboard the Titanic, we predict whether they survived (1) or did not survive (0).
We use real historical passenger data (age, sex, ticket class, etc.) and train a Random Forest Classifier — an ensemble machine learning model — to learn patterns from the data and make predictions.
This is one of the most famous beginner ML competitions on Kaggle and is considered the "Hello World" of machine learning.
In plain English: We're teaching a computer to recognize survival patterns from historical passenger records so it can predict outcomes for passengers it has never seen before.

🔄 The Machine Learning Pipeline — Big Picture
Before diving into code, understand the overall flow:
Raw Data (CSV files)
       ↓
Exploratory Data Analysis (What does the data look like?)
       ↓
Data Cleaning (Fix missing values, remove useless columns)
       ↓
Feature Engineering (Convert text → numbers)
       ↓
Train/Validation Split (80% train, 20% validate)
       ↓
Model Training (Random Forest learns patterns)
       ↓
Model Evaluation (How accurate is it on data it hasn't seen?)
       ↓
Final Prediction (Predict on test.csv)
       ↓
Submission File (submission.csv → upload to Kaggle)
Every ML project follows roughly this flow. Master it here and you'll recognize it in every future project.

⚙️ Setup & Requirements
Running on Kaggle (Recommended)
This notebook is designed to run on Kaggle Notebooks, where the dataset is pre-loaded. No installation needed.

Go to kaggle.com/competitions/titanic
Click "Code" → "New Notebook"
Paste or upload the notebook
Click "Run All"

Running Locally
If you want to run it on your own machine:
Prerequisites:
bashpip install numpy pandas scikit-learn matplotlib seaborn
Download the data:
bashkaggle competitions download -c titanic
unzip titanic.zip

You'll need the Kaggle CLI installed for this.

Update the file paths in the notebook:
python# Change this:
train_data = pd.read_csv("/kaggle/input/competitions/titanic/train.csv")

# To this (if running locally):
train_data = pd.read_csv("train.csv")
Library Versions (what this was built with)
LibraryPurposenumpyNumerical computationspandasData loading and manipulationscikit-learnML model, preprocessing, evaluationmatplotlibPlotting graphsseabornBeautiful statistical visualizations

📋 Step-by-Step Code Walkthrough

Step 1 — Importing Libraries
pythonimport numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
What's happening here?
Think of this as gathering your tools before starting a job. Each import brings in a toolkit:

numpy — Handles fast math on arrays/matrices. The backbone of scientific Python.
pandas — Your spreadsheet inside Python. Loads CSVs, lets you filter/sort/group rows. The DataFrame object is your best friend.
sklearn (scikit-learn) — The ML library. RandomForestClassifier is the model, train_test_split divides your data, LabelEncoder converts text to numbers, accuracy_score tells you how well the model did.
matplotlib + seaborn — Drawing charts. Seaborn is built on top of matplotlib and produces prettier statistical plots with less code.
sns.set_style("whitegrid") — Just makes plots look cleaner (white background with light grid lines).


💡 Beginner Tip: You don't need to memorize all these libraries. Just know what category each belongs to. You'll naturally remember them through use.


Step 2 — Loading the Data
train_data = pd.read_csv("/kaggle/input/competitions/titanic/train.csv")
test_data  = pd.read_csv("/kaggle/input/competitions/titanic/test.csv")
What's happening here?
pd.read_csv() reads a CSV file and turns it into a DataFrame — imagine it as a table (like Excel) stored in memory. Two separate DataFrames are created: one for training and one for testing.
Run train_data in a cell to preview it — Kaggle/Jupyter will display it as an interactive table.

Why two files? In real ML, you train on known data and predict on unknown data. train.csv has the Survived column (the answer). test.csv does not — that's what you need to predict.


Step 3 — Exploring the Data (EDA)
print(train_data.shape)   # (891, 12) — 891 rows, 12 columns
print(test_data.shape)    # (418, 11) — 418 rows, 11 columns (no 'Survived')
train_data.info()         # Column names, data types, non-null counts
train_data.isnull().sum() # Count of missing values per column
What's happening here?
EDA (Exploratory Data Analysis) is the detective phase. Before building anything, you need to understand your data.

.shape — tells you how many rows and columns exist
.info() — shows each column's data type and how many values are non-null (not missing)
.isnull().sum() — counts missing values column-by-column

What you'll typically see:
Age        177  ← 177 passengers have no age recorded
Cabin      687  ← Most cabin data is missing
Embarked     2  ← Just 2 passengers missing this
This tells you which columns need fixing before training.

💡 Beginner Tip: Never skip EDA. Rushing into model training without understanding your data is the #1 beginner mistake. Garbage in = garbage out.


Step 4 — Handling Missing Values
# Fill missing Ages with the median age
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age']  = test_data['Age'].fillna(test_data['Age'].median())

# Fill missing Fare in test data with median
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# Fill missing Embarked with the most common port
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
What's happening here?
Machine learning models cannot work with missing (NaN) values. You must handle them first.
There are several strategies. This notebook uses imputation — filling in missing values with a sensible substitute:
ColumnStrategy UsedWhy This ChoiceAgeMedianAge has outliers (very old/young passengers). Median is more robust than mean in such cases.FareMedianSimilar reasoning — fare has high outliers (luxury suites).EmbarkedMode (most frequent)Categorical value. "Most common port" is the simplest logical fill.
Why median and not mean?

Mean is sensitive to extreme values. If 5 people paid £500 and 100 people paid £7, the mean gets dragged up. Median just picks the middle value — it's more "typical."


💡 Beginner Tip: There's no single "right" way to handle missing values. Other approaches include: dropping rows with missing data (dropna()), predicting the missing value using another model, or creating a new "Unknown" category. The best approach depends on how much data is missing and how important the column is.


Step 5 — Dropping Irrelevant Columns
python# Drop Name and Ticket — not useful as raw text for prediction
train_data.drop(['Name', 'Ticket'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Drop Cabin — too many missing values to be useful
train_data.drop('Cabin', axis=1, inplace=True, errors='ignore')
test_data.drop('Cabin', axis=1, inplace=True, errors='ignore')
What's happening here?
Not every column is useful for prediction. Some columns are dropped because:
ColumnReason DroppedNameEach name is unique — a model can't generalize from it. (More advanced projects extract titles like "Mr.", "Mrs." from names as a feature.)TicketTicket numbers are semi-random strings with no clear pattern.Cabin77% of values are missing. Not enough data to be useful.
What axis=1 means:

axis=0 = operate on rows
axis=1 = operate on columns

So drop(..., axis=1) removes a column. drop(..., axis=0) would remove a row.
What inplace=True means:

Normally pandas returns a new DataFrame. inplace=True modifies the original directly instead.
Without it: train_data = train_data.drop(...) ← you'd need to reassign.


💡 Beginner Tip: errors='ignore' is a safety net — if the column doesn't exist, Python won't crash. Useful when the same code runs on both train and test data which may differ slightly.


Step 6 — Encoding Categorical Variables
pythonle_sex = LabelEncoder()
train_data['Sex'] = le_sex.fit_transform(train_data['Sex'])
test_data['Sex']  = le_sex.transform(test_data['Sex'])

le_embarked = LabelEncoder()
train_data['Embarked'] = le_embarked.fit_transform(train_data['Embarked'])
test_data['Embarked']  = le_embarked.transform(test_data['Embarked'])
What's happening here?
ML models work with numbers, not text. Columns like Sex ("male"/"female") and Embarked ("S"/"C"/"Q") need to be converted.
LabelEncoder assigns each unique text value an integer:

female → 0, male → 1
C → 0, Q → 1, S → 2

Critical distinction — fit_transform vs transform:
MethodUsed OnWhat It Doesfit_transformTraining dataLearns the mapping ("female"=0, "male"=1) AND applies ittransformTest dataOnly applies the already-learned mapping — does NOT re-learn
Why? If you called fit_transform on both, the test data might get a different mapping — the model learned "female=0" but now test data maps "female=1". Predictions would be completely wrong.

💡 Beginner Tip: Always fit on train data, then transform both. This rule applies to all preprocessing in scikit-learn (scalers, encoders, etc.). It prevents data leakage — accidentally letting test data influence how you preprocess.


Step 7 — Splitting Features and Target
pythonX = train_data.drop(['Survived', 'PassengerId'], axis=1)  # Features
y = train_data['Survived']                                  # Target
What's happening here?
In supervised ML, every row has:

Features (X): The input variables — what we know about each passenger
Target (y): What we're trying to predict — whether they survived

Think of it as: X is the question, y is the answer.
PassengerId is also dropped from X because it's just an arbitrary ID number — it carries no meaningful signal about survival.
After this step, X looks like:
   Pclass  Sex  Age  SibSp  Parch   Fare  Embarked
0       3    1   22      1      0   7.25         2
1       1    0   38      1      0  71.28         0
...
And y looks like:
0    0
1    1
...

Step 8 — Train-Validation Split
pythonX_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
What's happening here?
We have 891 labeled passengers in train.csv. We split them:
Full training data (891 passengers)
         ↓
┌─────────────────────┬────────────────┐
│  X_train / y_train  │ X_valid/y_valid │
│    80% = 712 rows   │  20% = 179 rows │
│  (model trains on)  │  (model tested  │
│                     │   before Kaggle)│
└─────────────────────┴────────────────┘
Why do this?
If you trained on all 891 rows and evaluated on the same 891 rows, you'd get deceptively high accuracy — the model just memorized the data. This is called overfitting.
The validation split gives you an honest estimate of how well the model will perform on truly unseen data.
Parameters explained:

test_size=0.2 → 20% goes to validation
random_state=42 → Fixed random seed so the split is the same every time you run. Without this, each run splits differently and results aren't reproducible. (42 is a popular convention — any number works.)


Step 9 — Training the Random Forest Model
pythonmodel = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)
What's happening here?
This is where the machine actually learns.
What is a Random Forest?
A Random Forest is an ensemble of Decision Trees. Here's the intuition:

Decision Tree — A flowchart of yes/no questions:

   Is the passenger female?
   ├── Yes → Did they pay more than £30?
   │         ├── Yes → SURVIVED
   │         └── No  → SURVIVED
   └── No  → Was their Pclass 1?
             ├── Yes → SURVIVED
             └── No  → DID NOT SURVIVE

Random Forest — Instead of one tree, build 200 trees (n_estimators=200), each trained on a random subset of data and features. Then take a majority vote across all 200 trees.
Why? One tree overfits easily. 200 trees that each see slightly different data tend to average out each other's errors → better generalization.

Key Parameters:
ParameterValueMeaningn_estimators200Build 200 decision treesmax_depth5Each tree can ask at most 5 questions deep. Prevents overfitting.random_state42Reproducibility
.fit(X_train, y_train) is where learning happens. The model sees features and labels together and learns which feature patterns predict survival.

💡 Beginner Tip: You don't need to understand every math detail behind Random Forests to use them effectively. The key insight is: many slightly different trees voting together beats one perfect-looking tree.


Step 10 — Evaluating the Model
pythonpred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, pred)
print("Accuracy:", accuracy)
What's happening here?
model.predict(X_valid) makes predictions on the validation set — the 20% of passengers the model never trained on.
accuracy_score(y_valid, pred) compares those predictions against the actual known outcomes.
Formula:
Accuracy = (Number of correct predictions) / (Total predictions)
If the model correctly identifies 155 out of 179 validation passengers:
Accuracy = 155 / 179 ≈ 0.866 → 86.6%
Is 80-85% good?
For the Titanic dataset with this approach: yes, it's a solid baseline. Top Kaggle scores reach ~83-85% on the test set, so if your validation accuracy is in this range, you're doing well.

💡 Beginner Tip: Accuracy is not always the best metric. For imbalanced datasets (e.g., 95% of people didn't survive), a model that always predicts "not survived" would have 95% accuracy but be useless. For Titanic, the split is roughly 38% survived / 62% didn't — not severely imbalanced, so accuracy is fine here.


Step 11 — Making Predictions on Test Data
python# Retrain on ALL training data (including validation portion)
model.fit(X, y)

# Prepare test features
X_test = test_data.drop('PassengerId', axis=1)

# Make predictions
predictions = model.predict(X_test)
What's happening here?
Before predicting on test.csv, we retrain on the full training set (all 891 rows, not just the 80% split).
Why retrain?

During evaluation, we held out 20% to get an honest accuracy estimate
Now that we've validated the model is good, there's no reason to hold back that 20%
More training data → better model → better predictions on Kaggle

predictions is now an array of 0s and 1s — one for each of the 418 test passengers.

Step 12 — Generating the Submission File
pythonsubmission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})

submission.to_csv('/kaggle/working/submission.csv', index=False)

print("submission.csv created successfully!")
print(os.listdir('/kaggle/working'))  # Verify it exists

submission.head()  # Preview first 5 rows
What's happening here?
Kaggle competitions require a specific submission format:
PassengerId,Survived
892,0
893,1
894,0
...
We create this as a DataFrame and save it as a CSV. index=False prevents pandas from writing an extra numbered column at the start.
To submit: in your Kaggle notebook, click "Submit" in the top-right after running, and it will detect submission.csv automatically.

Step 13 — Visualizations
The notebook includes several visualizations to understand the data. Here's what each one reveals:
1. Survival Distribution
pythonsns.countplot(x='Survived', data=train_data)
plt.title("Survival Distribution")
Shows overall class balance: ~38% survived, ~62% didn't.
2. Survival by Gender
pythonsns.countplot(x='Sex', hue='Survived', data=train_data)
plt.title("Male vs Female Survival")
Reveals the single biggest survival predictor: females survived at ~74%, males at ~19%. This is the "women and children first" policy in action.
3. Survival by Passenger Class
pythonsns.countplot(x='Pclass', hue='Survived', data=train_data)
plt.title("Survival by Passenger Class")
First-class passengers had a ~63% survival rate. Third-class had ~24%. Wealth and access to lifeboats were correlated.
4. Age Distribution
pythonsns.histplot(train_data['Age'], bins=30, kde=True)
plt.title("Age Distribution")
Shows most passengers were 20–40 years old. The KDE (Kernel Density Estimate) is the smooth curve over the bars — it estimates the probability distribution.
5. Fare Distribution
pythonsns.histplot(train_data['Fare'], bins=40, kde=True)
plt.title("Fare Distribution")
Right-skewed — most passengers paid low fares, but a few paid extremely high amounts (first-class luxury cabins).
6. Correlation Heatmap
pythonsns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
Shows how strongly each pair of columns is correlated:

Values close to +1 = strong positive correlation
Values close to -1 = strong negative correlation
Values close to 0 = no linear relationship

Key finding: Sex and Pclass have the highest correlation with Survived.
7. Age & Fare vs Survival (Box Plots)
pythonsns.boxplot(x='Survived', y='Age', data=train_data)
sns.boxplot(x='Survived', y='Fare', data=train_data)
Box plots show the distribution within each survival group. Survivors tended to be slightly younger and paid notably higher fares.

💡 Beginner Tip: EDA visualizations are not just for pretty charts — they tell you which features are likely to matter most for your model. Always look at your data before tuning hyperparameters.


Step 14 — Feature Importance
pythonimportance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance)

sns.barplot(x='Importance', y='Feature', data=importance)
plt.title("Feature Importance")
What's happening here?
After training, Random Forest can tell you which features it relied on most to make decisions. model.feature_importances_ returns a score for each feature — higher = more important.
Typical output for Titanic:
    Feature   Importance
      Sex        0.28
      Fare       0.24
      Age        0.22
    Pclass       0.15
    ...
This confirms what the EDA visualizations suggested: Sex, Fare, and Age are the strongest predictors of survival.

💡 Beginner Tip: Feature importance is a powerful debugging tool. If a feature you expected to matter isn't in the top results, either the feature genuinely doesn't help (remove it) or it wasn't preprocessed properly (fix it).


📈 Results & Output
MetricValueValidation Accuracy~82–86% (typical for this setup)Model UsedRandom Forest (200 trees, max depth 5)Output Filesubmission.csvSubmission FormatPassengerId, Survived
Top features by importance:

Sex (gender)
Fare (ticket price)
Age
Pclass (ticket class)


🧩 Key Concepts Explained Simply
Classification — Predicting which category something belongs to. Here: survived (1) or not (0). As opposed to regression, which predicts a continuous number.
Overfitting — When a model memorizes the training data instead of learning general patterns. It scores 99% on training data but poorly on new data. max_depth=5 in our Random Forest limits this.
Underfitting — The opposite: the model is too simple and fails to capture real patterns even in training data.
Label Encoding — Converting text categories to integers. female → 0, male → 1.
Imputation — Filling in missing values using a statistical substitute (mean, median, mode) rather than deleting rows.
Ensemble Method — Combining multiple models (here: many trees) so their combined prediction is more accurate and stable than any single model.
Train/Validation Split — Dividing your labeled data so some is used for learning and some is held back for honest evaluation.

🐛 Common Errors & Fixes
ValueError: Input contains NaN

You forgot to fill missing values in a column before training.
Fix: Run .isnull().sum() on your feature matrix and fill any remaining nulls.

KeyError: 'Cabin'

You're trying to drop a column that was already dropped.
Fix: The code already handles this with errors='ignore'. Make sure you're running cells in order.

ValueError: could not convert string to float

A text column wasn't encoded before training.
Fix: Check X_train.select_dtypes(include=['object']).columns to find remaining string columns and encode them.

Predictions all 0 or all 1

Model might be biased. Check class balance with y.value_counts().
Consider adjusting class_weight='balanced' in RandomForestClassifier.

File not found at /kaggle/input/...

You're running locally but the path points to Kaggle.
Fix: Change paths to where you've saved the CSV files.


🚀 How to Improve This Further
Once you understand the baseline, here are next steps to push your accuracy higher:

Feature Engineering

Extract titles from names: "Mr.", "Mrs.", "Miss.", "Master" → strong survival signal
Create a FamilySize column = SibSp + Parch + 1
Create an IsAlone binary feature


Better Missing Value Handling

Predict missing Age using a regression model based on Pclass and Sex
Use Cabin deck (first letter) instead of dropping entirely


Try Other Models

Logistic Regression (simpler, interpretable)
Gradient Boosting (XGBoost / LightGBM) — often outperforms Random Forest on tabular data
Support Vector Machines


Hyperparameter Tuning

Use GridSearchCV or RandomizedSearchCV to find optimal n_estimators and max_depth


Cross-Validation

Instead of a single 80/20 split, use cross_val_score with k=5 folds for a more reliable accuracy estimate


Ensemble Multiple Models

Average predictions from Random Forest + Logistic Regression + XGBoost for a stronger final prediction


Built with Python · scikit-learn · pandas · seaborn
