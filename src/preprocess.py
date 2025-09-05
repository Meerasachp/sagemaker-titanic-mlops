import pandas as pd

# Load dataset
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Fill missing values
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Encode categorical variables
train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)

# Separate features and target
X_train = train.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
y_train = train['Survived']
X_test = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# Save preprocessed data
X_train.to_csv("data/X_train.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

