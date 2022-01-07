# Readme

Data wrangling notebooks:

## Intro

**Armenian Online Job Postings**

The online job market is a good indicator of overall demand for labor in an economy. This dataset consists of 19,000 job postings from 2004 to 2015 posted on CareerCenter, an Armenian human resource portal.

Since postings are text documents and tend to have similar structures, text mining can be used to extract features like posting date, job title, company name, job description, salary, and more. Postings that had no structure or were not job-related were removed. The data was originally scraped from a Yahoo! mailing group

## Gathering data

**Best of Rotten Tomatoes**

How do you pick which movie to watch? You may check movie rating websites like Rotten Tomatoes or IMDb to help you chose. These sites contain a number of different metrics which are used to evaluate whether or not you will like a movie. However, because these metrics do not always show on the same page, figuring out the best movies can get confusing.

We can start with the [Rotten Tomatoes: Top 100 Movies of All Time](https://www.rottentomatoes.com/top/bestofrt/). (Note: this current list may be different than the latest archived list used in this lesson). We can use a scatter plot to look at audience scores vs critics scores for each movie.

For lots of people, Roger Ebert's movie review was the only review they needed because he explained the movie in such a way that they would know whether they would like it or not. Wouldn't it be neat if we had a word cloud like this one for each of the movies in the top 100 list at [RogerEbert.com](http://www.rogerebert.com/)? We can use a [Andreas Mueller's Word Cloud Generator](https://amueller.github.io/word_cloud/) in Python to help.

## Assessing data

**Oral Insulin Phase II Clinical Trial**

We will be looking at the phase two clinical trial data of 350 patients for a new innovative oral insulin called Auralin - a proprietary capsule that can solve this stomach lining problem.

Phase two trials are intended to:

- Test the efficacy and the dose response of a drug
- Identify adverse reactions

In this trial, half of the patients are being treated with Auralin, and the other 175 being treated with a popular injectable insulin called Novodra. By comparing key metrics between these two drugs, we can determine if Auralin is effective. 

*Why do we need Data Cleaning?*

Healthcare data is notorious for its errors and disorganization, and its clinical trial data is no exception. For example, human errors during the patient registration process means we can have

- duplicate data
- missing data
- inaccurate data

We're going to take the first step in fixing these issues by assessing this data sets quality and tidiness, and then cleaning all of these issues using Python and Pandas. Our goal is to create a trustworthy analysis.

*DISCLAIMER: This Data Isn't "Real"*

The Auralin and Novodra are not real insulin products. This clinical trial data was fabricated for the sake of this course. When assessing this data, the issues that you'll detect (and later clean) are meant to simulate real-world data quality and tidiness issues.

That said:

- This dataset was constructed with the consultation of real doctors to ensure plausibility.
- This clinical trial data for an alternative insulin was inspired and closely mimics this real [clinical trial for a new inhaled insulin called Afrezza](http://care.diabetesjournals.org/content/38/12/2266.long).
- The data quality issues in this dataset mimic real, [common data quality issues in healthcare data](http://media.hypersites.com/clients/1446/filemanager/Articles/DocCenter_Problem_with_data.pdf). These issues impact quality of care, patient registration, and revenue.
- The patients in this dataset were created using this [fake name generator](http://www.fakenamegenerator.com/order.php) and do not include real names, addresses, phone numbers, emails, etc.

## Cleaning data

