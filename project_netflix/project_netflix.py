# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:13:32 2021

@author: daria
"""

# PROJECT: Netflix Movies and TV Shows

# IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# EVALUATING DATA

# Generating a DataFrame
df = pd.read_csv("netflix_titles.csv")

# Size of the DataFrame
print(df.shape) # dataset consists of 7.787 rows and 12 columns

# Description of the DataFrame
print(df.info())

# Information as to the DataFrame
print(df.head())

# Row and column labels of the DataFrame
print(df.index)
print(df.columns)

# Descriptive Data Analysis of the column "release year"
print(df.describe()) # movies/TV shows produced range from 1925 to 2021

# Counts of unique values and most frequently-occuring elements
print(df["show_id"].value_counts())
print(df["type"].value_counts(normalize=True)) # Movie most frequently-occuring
print(df["title"].value_counts())
print(df["director"].value_counts())
print(df["cast"].value_counts())
print(df["country"].value_counts(normalize=True)) # United States most frequently-occuring
print(df["date_added"].value_counts())
print(df["release_year"].value_counts(normalize=True)) # 2018 most frequently-occuring
print(df["rating"].value_counts(normalize=True)) # TV-MA most frequently-occuring
print(df["duration"].value_counts(normalize=True)) # 1 season most frequently-occuring
print(df["listed_in"].value_counts(normalize=True)) # Documentaries most frequently-occuring
print(df["description"].value_counts())



# CLEANING AND PREPARING DATA

# Identifying zero values
print(df.isnull())

# Summing up to look at the number of zero values
print(df.isnull().sum())

# Eliminating data with missing values
print(df.dropna(subset=["country"], inplace=True))
print(df.dropna(subset=["date_added"], inplace=True))
print(df.dropna(subset=["rating"], inplace=True))

# Replacing zero values with "missing" of the remaining two columns
print(df["director"].fillna(value="missing", inplace=True))
print(df["cast"].fillna(value="missing", inplace=True))

# Checking again for zero values and looking at the actual size
print(df.isnull().sum())
print(df.shape) # New number of rows: 7.265

# Renaming some columns
df = df.rename(columns={"listed_in": "genre"})
df = df.rename(columns={"release_year": "release year"})
print(df.columns)




# VISUALIZATION

# What is the most frequent type of content on Netflix
sns.set(rc={"figure.figsize":(15,8)})
sns.countplot(x="type", data=df, palette="spring")
plt.title("Type of content on Netflix")
plt.savefig("type.png", dpi=100)
plt.show()



# Which country released the majority of the movies/TV shows
# Creating "Top 15" of the column "country" for better visualization
count_country = df["country"].value_counts().sort_values(ascending=False)
country15 = count_country.head(15)
sns.set(rc={"figure.figsize":(15,8)})
sns.barplot(x=country15.values, y=country15.index)
plt.title("Top 15 content producing countries on Netflix")
plt.savefig("country.png", dpi=100)
plt.show()



# When were produced the majority of the movies/TV shows
# Filtering the column "release year" for better visualization
df_year = df[df["release year"] >= 2000]
sns.set(rc={"figure.figsize":(12,4)})
sns.countplot(x="release year", data=df_year)
plt.title("Content released annually from 2000 to 2021")
plt.savefig("year.png", dpi=100)
plt.show()

# Setting column "type" as hue 
df_year = df[df["release year"] >= 2000]
sns.set(rc={"figure.figsize":(12,4)})
sns.countplot(x="release year", data=df_year, hue="type", palette="cool")
plt.title("Content released annually (2000-2021) by type")
plt.savefig("year_type.png", dpi=100)
plt.show()



# What is the distribution of television content rating
# Setting column "type" as hue
sns.set(rc={"figure.figsize":(12,4)})
sns.countplot(x="rating", data=df, hue="type", palette="Oranges")
plt.title("Distribution of movie and TV show rating")
plt.savefig("tvcr_type.png", dpi=100)
plt.show()



# What is the duration of movies and TV shows
# Adjusting the DataFrame by splitting the column "duration"
df[['duration','unit']] = df["duration"].str.split(" ", 1, expand=True)
print(df.head())

# Distribution of movie duration
df_movie = df[df["type"]=="Movie"]
sns.set(rc={"figure.figsize":(15,8)})
sns.distplot(df_movie["duration"], kde=False, bins=50, color="c")
plt.title("Distribution of movie duration")
plt.xlim(0,200)
plt.savefig("dur_movie.png", dpi=100)
plt.show()

# Distribution of TV show duration
df_tvshow = df[df["type"]=="TV Show"]
sns.set(rc={"figure.figsize":(15,8)})
sns.distplot(df_tvshow["duration"], kde=False, bins=20, color="m")
plt.title("Distribution of TV show duration")
plt.xlim(1,10)
plt.savefig("dur_tvshow.png", dpi=100)
plt.show()



# Which genre is the most popular on Netflix
# Filtering DataFrame by column "genre" for movie
movies = df[df["type"] == "Movie"]
sns.set(rc={"figure.figsize":(25,10)})
sns.barplot(x=movies["genre"].value_counts().sort_values(ascending=False).head(10).index,
y=movies["genre"].value_counts().sort_values(ascending=False).head(10).values,
palette=("spring"))
plt.xticks(rotation=10)
plt.title("Top 10 Neflix movie genre")
plt.savefig("movie10genre.png", dpi=100)
plt.show()

# Filtering DataFrame by column "genre" for TV show
tvshow = df[df['type'] == 'TV Show']
sns.set(rc={"figure.figsize":(25,10)})
sns.barplot(x=tvshow["genre"].value_counts().sort_values(ascending=False).head(10).index,
y=tvshow["genre"].value_counts().sort_values(ascending=False).head(10).values,
palette=("cool"))
plt.xticks(rotation=10)
plt.title("Top 10 Neflix TV show genre")
plt.savefig("tvshow10genre.png", dpi=100)
plt.show()

