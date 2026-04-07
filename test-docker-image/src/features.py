import pandas as pd


NUMERIC_FEATURES = ["Age", "Fare", "FarePerPerson", "FamilySize", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone"]

RARE_TITLES = ["Capt", "Col", "Don", "Dona", "Dr", "Jonkheer",
               "Lady", "Major", "Mlle", "Mme", "Rev", "Sir", "Countess"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    df["Title"] = df["Title"].apply(lambda x: "Rare" if x in RARE_TITLES else x)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    return df
