import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\amarj\Downloads\laptopData.csv")


if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)


if "Ram" in df.columns:
    df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", regex=False)
    df["Ram"] = pd.to_numeric(df["Ram"], errors="coerce")


if "Weight" in df.columns:
    df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")


price_col = None

if "Price" in df.columns:
    price_col = "Price"
elif "Price_euros" in df.columns:
    price_col = "Price_euros"


if price_col:
    Q1 = df[price_col].quantile(0.25)
    Q3 = df[price_col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[price_col] < lower) |
        (df[price_col] > upper)
    ]

    print("\nOutliers Found:", len(outliers))


plt.figure(figsize=(10,5))
df["Company"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Laptop Brands")
plt.xlabel("Company")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


if price_col:
    plt.figure(figsize=(8,5))
    sns.histplot(df[price_col], bins=30, kde=True)
    plt.title("Price Distribution")
    plt.tight_layout()
    plt.show()


if "Ram" in df.columns and price_col:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="Ram", y=price_col, data=df)
    plt.title("RAM vs Price")
    plt.tight_layout()
    plt.show()


numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


if price_col:
    print("\nAverage Price by Company:")
    print(
        df.groupby("Company")[price_col]
        .mean()
        .sort_values(ascending=False)
    )