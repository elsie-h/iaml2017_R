Lab\_3\_SVM
================
Elsie Horne
8 December 2017

Lab 3: Support Vector Mahcines
==============================

1. Spam filtering
=================

Load tidyverse

``` r
library(tidyverse)
```

========== Question 1.1 ==========
----------------------------------

**Load spambase\_binary.csv, display the number of instances and attributes and the first 5 samples. Remember that the attributes have been binarised. The instances have also been shuffled (i.e. their order has been randomised)**

``` r
spambase_binary <- read.csv(file = "spambase_binary.csv") # need to sort out the encoding here for the special characters
spambase_binary <- as_tibble(spambase_binary)
dim(spambase_binary)[1] # instances
```

    ## [1] 4601

``` r
dim(spambase_binary)[2] # attributes
```

    ## [1] 55

``` r
head(spambase_binary, 5)
```

    ## # A tibble: 5 x 55
    ##   word_freq_make_binarized word_freq_address_binarized
    ##                      <int>                       <int>
    ## 1                        0                           1
    ## 2                        0                           0
    ## 3                        0                           0
    ## 4                        0                           0
    ## 5                        0                           0
    ## # ... with 53 more variables: word_freq_all_binarized <int>,
    ## #   word_freq_3d_binarized <int>, word_freq_our_binarized <int>,
    ## #   word_freq_over_binarized <int>, word_freq_remove_binarized <int>,
    ## #   word_freq_internet_binarized <int>, word_freq_order_binarized <int>,
    ## #   word_freq_mail_binarized <int>, word_freq_receive_binarized <int>,
    ## #   word_freq_will_binarized <int>, word_freq_people_binarized <int>,
    ## #   word_freq_report_binarized <int>, word_freq_addresses_binarized <int>,
    ## #   word_freq_free_binarized <int>, word_freq_business_binarized <int>,
    ## #   word_freq_email_binarized <int>, word_freq_you_binarized <int>,
    ## #   word_freq_credit_binarized <int>, word_freq_your_binarized <int>,
    ## #   word_freq_font_binarized <int>, word_freq_000_binarized <int>,
    ## #   word_freq_money_binarized <int>, word_freq_hp_binarized <int>,
    ## #   word_freq_hpl_binarized <int>, word_freq_george_binarized <int>,
    ## #   word_freq_650_binarized <int>, word_freq_lab_binarized <int>,
    ## #   word_freq_labs_binarized <int>, word_freq_telnet_binarized <int>,
    ## #   word_freq_857_binarized <int>, word_freq_data_binarized <int>,
    ## #   word_freq_415_binarized <int>, word_freq_85_binarized <int>,
    ## #   word_freq_technology_binarized <int>, word_freq_1999_binarized <int>,
    ## #   word_freq_parts_binarized <int>, word_freq_pm_binarized <int>,
    ## #   word_freq_direct_binarized <int>, word_freq_cs_binarized <int>,
    ## #   word_freq_meeting_binarized <int>, word_freq_original_binarized <int>,
    ## #   word_freq_project_binarized <int>, word_freq_re_binarized <int>,
    ## #   word_freq_edu_binarized <int>, word_freq_table_binarized <int>,
    ## #   word_freq_conference_binarized <int>, char_freq_._binarized <int>,
    ## #   char_freq_._binarized.1 <int>, char_freq_._binarized.2 <int>,
    ## #   char_freq_._binarized.3 <int>, char_freq_._binarized.4 <int>,
    ## #   char_freq_._binarized.5 <int>, is_spam <int>

========== Question 1.2 ==========
----------------------------------

**We are going to use hold-out validation to evaluate our models below. Split the dataset into training and testing subsets. Use 90% of the data for training and the remaining 10% for testing.**

**If you want to be able to reproduce your results exactly, what argument must you remember to set?** Must set a seed.

``` r
set.seed(123) # set seed so that we produce the same train/test sets if analysis repeated

spambase_binary <- spambase_binary %>% # create variables to split to train and test set
  mutate(train = runif(nrow(spambase_binary)), rank = rank(train)) 

spam_test  <- spambase_binary %>% # create test set
  filter(rank<=0.1*nrow(spambase_binary)) %>%
  select(-train, - rank)

spam_train <- spambase_binary %>% # create train set
  filter(rank>0.1*nrow(spambase_binary)) %>%
  select(-train, - rank)

# check the resulting datasets
dim(spam_train)
```

    ## [1] 4141   55

``` r
dim(spam_test)
```

    ## [1] 460  55

========== Question 1.3 ==========
----------------------------------

Train a LogisticRegression classifier by using training data. Report the classification accuracy on both the training and test sets. Does your classifier generalise well on unseen data?

``` r
lr_model <- glm(is_spam ~ ., data = spam_train, family = "binomial")
# train set predictions and accuracy
spam_train$spam_prob <- predict(lr_model, type = "response")
spam_train$spam_pred <- ifelse(spam_train$spam_prob > mean(spam_train$is_spam), 1, 0)
(acc_train <- round(mean(spam_train$spam_pred == spam_train$is_spam), 2))
```

    ## [1] 0.94

``` r
# test set predictions and accuracy
spam_test$spam_prob <- predict(lr_model, newdata = spam_test, type = "response")
spam_test$spam_pred <- ifelse(spam_test$spam_prob > mean(spam_train$is_spam), 1, 0) # use the mean from the train set
(acc_test <- round(mean(spam_test$spam_pred == spam_test$is_spam), 2))
```

    ## [1] 0.93

The accuracy on the train set is 0.94 and the accuracy on the test set is 0.93, these are close, suggesting the model generalises well to new data.

========== Question 1.4 ==========
----------------------------------

\*\*Print the coefficients for class 1 for the attributes word\_freq\_hp\_binarized and char\_freq\_$\_binarized.\*\*

An encoding error means that `char_freq_$_binarized` appears as `char_freq_._binarized.4` - I tried to sort this out with the `fileEndcoing` and `encoding` options in `read.csv` but couldn't work it out. I will look into this when I have more time.

``` r
(lo_hp <- round(lr_model$coefficients["word_freq_hp_binarized"], 2))
```

    ## word_freq_hp_binarized 
    ##                  -3.14

``` r
(lo_.4 <- round(lr_model$coefficients["char_freq_._binarized.4"], 2))
```

    ## char_freq_._binarized.4 
    ##                    2.06

**Generally, we would expect the string $ to appear in spam, and the string hp to appear in non-spam e-mails, as the data was collected from HP Labs. Do the regression coefficients make sense given that class 1 is spam? Hint: Consider the sigmoid function and how it transforms values into a probability between 0 and 1. Since our attributes are boolean, a positive coefficient can only increase the total sum fed through the sigmoid and thus move the output of the sigmoid towards 1. What can happen if we have continuous, real-valued attributes?**

Interpretation f coefficients: The log odds of an email being spam changes by -3.14 and 2.06 when "hp" and "$" are present in the email respectively. I.e. presence of "hp" makes the email less likely to categorized as spam, "$" more likey to be categorized as spam. This makes sense.

Note1: I tried to name the variable "lo\_$", which was fine using backticks when assigning the name, but couldn't work out how to print this in R markdown without producing errors.

Note2: Come back to this lab after doing and SVM tutorial.
