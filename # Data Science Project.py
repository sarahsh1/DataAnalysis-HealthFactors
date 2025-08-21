# Data Science Project 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 

# Loading csv file of dataset and storing it in variable which can 
# be used later in the program
data = pd.read_csv("StressLevelDataset.csv")

# DATA CLEANING: 
# identify if any data is missing
print(data.isnull().sum())

# Print first 5 rows of dataset to
# test if data is correctly loaded
# print(data.head())

#DATA CLEANING: 
# identify and remove outliers
outliers_index = []

for column in data:
    # Calculate inter quartile range
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    for index in outliers.index:
        if index not in outliers_index:
            outliers_index.append(index)

    #Print number of outliers in each column
    print(f"Number of outliers in {column}: {len(outliers)}")

#data = data.drop(outliers_index)

# Calculate number of students with mental health history 
# 1 is yes, 0 is no
mental_health = data['mental_health_history'].value_counts()
print(mental_health)

# average depression rate in students
depression_avg = data['depression'].mean()
print("Depression average = ", depression_avg)

# pie chart of students in diff noise levels
noise = data['noise_level'].value_counts()
print(noise)

noise.groupby(['noise_level']).sum().plot(  kind='pie', ylabel= "noise level", autopct='%1.0f%%') 

# Average anxiety level
avg_anx = data['anxiety_level'].mean()
print(avg_anx)

# Number of students with below avg self esteem 
low_self_esteem_count = data[data['self_esteem'] < 15].shape[0]
print("Students with below average self esteem: ", low_self_esteem_count)

# Bar chart displaying Headache scores 
headaches = data['headache'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
headaches.plot(kind='bar', color='lightpink', edgecolor='black')
plt.title("Frequency of Headache Scores")
plt.xlabel("Headache Score")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)  # Keep the x-axis labels horizontal
plt.show()

# Bar chart displaying sleep quality
sleep_quality = data['sleep_quality'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
sleep_quality.plot(kind='bar', color='lightblue', edgecolor='black')
plt.title("Sleep quality of Students")
plt.xlabel("Sleep Quality")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)  # Keep the x-axis labels horizontal
plt.show()

# Number of students with below average academic performance 
low_academic_perf = data[data['academic_performance'] < 3].shape[0]
print("Students with below average academic performance: ", low_academic_perf)


# overlapping histograms to show relation between sleep 
# quality and depression
plt.figure(figsize=(8, 6))

# Poor sleep quality
poor_sleep = data[data['sleep_quality'] < 3]
plt.hist(poor_sleep['depression'], bins=10, alpha=0.7, label="Poor Sleep Quality", color='red')

# Better sleep quality
good_sleep = data[data['sleep_quality'] >= 3]
plt.hist(good_sleep['depression'], bins=10, alpha=0.7, label="Good Sleep Quality", color='yellow')

plt.title("Depression Levels: Poor vs. Good Sleep Quality")
plt.xlabel("Depression Level")
plt.ylabel("Frequency")
plt.legend()
plt.show()


#FacetGrid to show correlation between depression and bullying levels
# Categorize bullying scores into bins
data['bullying_category'] = pd.cut(data['bullying'], bins=[0, 1, 2, 3, 4, 5], 
                                   labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

# Create a FacetGrid to show depression distribution across bullying categories
g = sns.FacetGrid(data, col='bullying_category', col_wrap=3, height=4, sharey=True)
g.map(sns.histplot, 'depression', bins=10, color='skyblue', kde=False)

# Add titles and labels
g.set_titles("Bullying Level: {col_name}")
g.set_axis_labels("Depression Level", "Frequency")
g.figure.subplots_adjust(top=0.9)
g.figure.suptitle("Depression Levels Across Bullying Categories", fontsize=16)

plt.show()

# Students who are involved in extracurricular activities
extra_activities = data[data['extracurricular_activities'] != 0].shape[0]
print("Students with extracurricular activities: ", extra_activities)

# Students who have been bullied
bullied = data[data['bullying'] != 0].shape[0]
print("Students who have been bullied: ", bullied)

# Students whose basic needs are unmet
needs = data[data['basic_needs'] == 0].shape[0]
print("Students whose basic needs are unmet: ", needs)