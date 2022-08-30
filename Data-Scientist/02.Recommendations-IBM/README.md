# Recommendation Engine with IBM

## Overview

In this project we build a recommendation engine with IBM data. We analyze the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles we think they will like.

The data is provided by [IBM Watson](https://www.ibm.com/watson). 

## Project Structure

### Exploratory Data Analysis

Before making recommendations of any kind, we will need to explore the data we are working with for the project.

### Rank Based Recommendations

We will first find the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles with the most interactions are the most popular. These are then the articles we might recommend to new users (or anyone depending on what we know about them).

### User-User Based Collaborative Filtering

In order to build better recommendations for the users of IBM's platform, we could look at users that are similar in terms of the items they have interacted with. These items could then be recommended to the similar users. This would be a step in the right direction towards more personal recommendations for the users.

### Matrix Factorization

Finally, we will complete a machine learning approach to building recommendations. Using the user-item interactions, we will build out a matrix decomposition. Using our decomposition, we will get an idea of how well we can predict new articles an individual might interact with.

We will finally discuss which methods we might use moving forward, and how we might test how well our recommendations are working for engaging users.
