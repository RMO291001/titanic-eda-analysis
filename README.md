# Titanic Dataset — EDA
---

Ribbie Mohammad Omar  
BSc in Electrical and Computer Engineering (ECE),  
RUET

**Date:** 12/04/26

---

<div style="page-break-after: always;"></div>

## Dataset Overview

**Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

| Property | Detail |
|---|---|
| **Rows** | 891 passengers |
| **Columns** | 12 features |
| **Target Variable** | `Survived` (0 = No, 1 = Yes) |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `PassengerId` | int | Unique ID (not a feature) |
| `Survived` | int | Target: 0 = Died, 1 = Survived |
| `Pclass` | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | str | Passenger name (not a feature) |
| `Sex` | str | Gender |
| `Age` | float | Age in years |
| `SibSp` | int | No. of siblings/spouses aboard |
| `Parch` | int | No. of parents/children aboard |
| `Ticket` | str | Ticket number (not a feature) |
| `Fare` | float | Passenger fare |
| `Cabin` | str | Cabin number (mostly missing) |
| `Embarked` | str | Port of embarkation (C / Q / S) |

---

<div style="page-break-after: always;"></div>

## Step 1 — Import Libraries & Load Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
```

---

## Step 2 — Initial Data Understanding

```python
df.head()     # First 5 rows
df.shape      # (rows, columns)
df.columns    # Column names
```

> **Rows** = number of passengers  
> **Columns** = features (Age, Fare, Pclass, etc.)

---

## Step 3 — Data Overview

```python
df.info()      # Data types + non-null counts
df.describe()  # Statistical summary (mean, std, min, max, etc.)
```

> `info()` → Shows data types and which columns have missing values  
> `describe()` → Shows statistical summary for **numeric** columns only

---

## Step 4 — Check Missing Values

```python
df.isnull().sum()
```

**Expected Output:**

| Column | Missing Count |
|---|---|
| `Age` | 177 |
| `Cabin` | 687 |
| `Embarked` | 2 |

> `Age` → ~20% missing — fill with **median**  
> `Cabin` → ~77% missing — **drop** the column  
> `Embarked` → only 2 missing — fill with **mode**

---

<div style="page-break-after: always;"></div>

## Step 5 — Handle Missing Values

### 1. Fill Age with Median

```python
df['Age'] = df['Age'].fillna(df['Age'].median())
```

> **Why median?** Median is robust to outliers. If a few passengers are very old/young, the mean gets pulled toward them — median stays stable.

### 2. Fill Embarked with Mode

```python
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

> **Why mode?** `Embarked` is a categorical column. Mode gives the most frequently occurring value, which is the best guess for 2 missing entries.

### 3. Drop Cabin (Too Many Missing)

```python
df.drop(columns=['Cabin'], inplace=True)
```

> 687 out of 891 rows are missing — over 77%. This column cannot be reliably imputed.

### ✅ Re-check

```python
df.isnull().sum()
```

> All values should now be **0**.

---

## Step 6 — Drop Irrelevant Columns

```python
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
```

> `PassengerId` and `Ticket` are unique identifiers — they carry no predictive value.  
> `Name` is free text — not useful for analysis unless you extract titles (advanced).

---

## Step 7 — Check & Handle Duplicates

```python
df.duplicated().sum()
```

```python
df.drop_duplicates(inplace=True)
```

---

<div style="page-break-after: always;"></div>

## Step 8 — Data Type Conversion

```python
df['Sex']      = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df['Pclass']   = df['Pclass'].astype('category')   # Pclass is ordinal, not numeric
```

> Converting to `category` saves memory and signals to analysis tools that these are **discrete labels**, not continuous numbers.

---

## Step 9 — Outlier Detection ⚠️ Important

### Boxplot for Fare

```python
sns.boxplot(x=df['Fare'])
plt.title("Fare Distribution — Outlier Check")
plt.show()
```

> Points far beyond the whiskers = **outliers**. Fare has extreme values (first-class passengers paid much more).

---

## Step 10 — Outlier Treatment (Capping / Winsorizing) ⚠️ Important

> ❌ **Do NOT remove Fare outliers by deleting rows.**  
> First-class passengers with high fares are *meaningful data points*, not noise.  
> Deleting them would remove important signal about survival.

**✅ Instead — Cap the outliers (Winsorizing):**

```python
Q1  = df['Fare'].quantile(0.25)
Q3  = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR

df['Fare'] = df['Fare'].clip(upper=upper_limit)
```

> `clip(upper=upper_limit)` replaces any value above the limit with the limit itself.  
> The **row is preserved** — only the extreme value is tamed.

---

<div style="page-break-after: always;"></div>

## Step 11 — Feature Engineering

### Family Size

```python
df['FamilySize'] = df['SibSp'] + df['Parch']
```

> Combines siblings/spouses and parents/children into a single feature.  
> Sets up the question: *"Does traveling with family affect survival?"*

### Is Alone

```python
df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
```

> 1 = Traveling alone, 0 = Traveling with family.  
> This is a common and powerful feature in Titanic ML models.

---

## Step 12 — Univariate Analysis

### Survival Count

```python
sns.countplot(x='Survived', data=df)
plt.title("Survival Count (0 = Died, 1 = Survived)")
plt.show()
```

> More passengers **died** than survived. This is an **imbalanced target variable** — important for ML later.

### Age Distribution

```python
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()
```

> `kde=True` draws a smooth density curve on top of the histogram.  
> Most passengers were between **20–40 years old**.

---

<div style="page-break-after: always;"></div>

## Step 13 — Bivariate Analysis

### Survival vs Gender

```python
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()
```

```python
print(df.groupby('Sex')['Survived'].mean().round(2))
```

**Survival Rates:**

| Gender | Survival Rate |
|---|---|
| Female | ~74% |
| Male | ~19% |

> **Insight:** Females survived at nearly 4× the rate of males — the strongest survival signal in this dataset.

---

### Survival vs Passenger Class

```python
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()
```

```python
print(df.groupby('Pclass')['Survived'].mean().round(2))
```

**Survival Rates:**

| Class | Survival Rate |
|---|---|
| 1st Class | ~63% |
| 2nd Class | ~47% |
| 3rd Class | ~24% |

> **Insight:** Higher class = higher survival. Social class heavily influenced who was given priority on lifeboats.

---

<div style="page-break-after: always;"></div>

## Step 14 — Correlation Analysis ⚠️ Important

> **Problem:** `df.corr()` silently skips categorical columns like `Sex` and `Embarked` — which are among the most important features.  
> **Fix:** Encode `Sex` numerically before plotting the heatmap.

```python
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})

sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```

> **Reading the heatmap:**  
> Values close to **+1** = strong positive relationship  
> Values close to **-1** = strong negative relationship  
> Values close to **0** = no linear relationship  
>
> `Sex_encoded` will show the strongest correlation with `Survived` — confirming our bivariate analysis.

---

## Quick Reference — Full EDA Checklist

| Step | Task | Key Function |
|---|---|---|
| 1 | Load data | `pd.read_csv()` |
| 2 | First look | `df.head()`, `df.shape` |
| 3 | Data overview | `df.info()`, `df.describe()` |
| 4 | Missing values | `df.isnull().sum()` |
| 5 | Impute missing | `.fillna(median / mode)` |
| 6 | Drop irrelevant cols | `df.drop(columns=[...])` |
| 7 | Duplicates | `df.duplicated().sum()` |
| 8 | Fix data types | `.astype('category')` |
| 9 | Detect outliers | `sns.boxplot()` |
| 10 | Treat outliers | `.clip(upper=...)` |
| 11 | Feature engineering | New columns from existing ones |
| 12 | Univariate analysis | `countplot`, `histplot` |
| 13 | Bivariate analysis | `countplot` + `groupby().mean()` |
| 14 | Correlation | `sns.heatmap(df.corr())` |

---

*End of Titanic EDA*
