# Prosper Loan Data
## by Natalia BOBKOVA


## Dataset

This [data set](https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv) contains 113,937 loans with 81 variables on each loan, including loan amount, borrower rate (or interest rate), current loan status, borrower income, and many others.

This [data dictionary](https://www.google.com/url?q=https://docs.google.com/spreadsheet/ccc?key%3D0AllIqIyvWZdadDd5NTlqZ1pBMHlsUjdrOTZHaVBuSlE%26usp%3Dsharing&sa=D&source=editors&ust=1643672087503009&usg=AOvVaw3Du8m8kG18JesrSzK2zqH3) explains the variables in the data set.


## Summary of Findings

In this exploration I was interested in finding out what variables are correlated to Prosper Score.

I had to perform some basic data cleaning in order to remove variables that are not well represented and to group variables.

Fist of all, I found out that the Prosper Score is almost normally distributed - it has the highest frequencies between the values 4 and 8, wheareas scores 1 and 11 are less numerous. This means that people tend to have moderate risk for their loans. There is also a correlation between Prosper Score and Loan Original amount - the higher the amount, the higher the score. As for correlation between Prosper Score and Employment status - prosper score is distributed more or less in the same way across different group, most of the data falls into current loan status for employed people. However, full-time employed people tend to take lower risks, whereas self-employed people - higher. 

Outside of the main variables of interest, I found other interesting trends in distributions of predictor variables. Overall, the data contain in majority current and completed loan status. Employed and full-time employed borrowers are also in majority (compared to self-employed, part-time employed, not employed and retired people). Loan original amount is low - the lower the amount, the more frequently it is borrowed. As for correlations, we found out that employment status is related to employment duration: not employed or part-time correspond to low employment status duration (short period of time unemployed or part-time), self-employed, employed, ful time - higher duration. It seems also that self-employed people are genarally not able to provide a proof of their employment. 


## Key Insights for Presentation

In my presentation I will focus on the main trends found in data, namely on two categorical variables (Prosper Score and Employment Status) and one numeric (Original Loan Amount). 

In Employment status data we encounter Not Available and Other values which can't be easily interpreted, we will drop them for presentation. We will also adjust bin sizes when visualizing numeric data. We will transform Prosper Score labels from float to integer for a better visialization effect.