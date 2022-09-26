# Imports
library(mltools)
library(data.table)
library(ggplot2)

library(kernlab)
library(ROCR)
library(e1071)

prepare_dataset <- function(labels, round) {
  # Load the dataset
  raw_labels <- data.frame(
    read.csv(
      labels,
      header = F,
      sep = " ",
      dec = "."
    )
  )
  raw_features <- data.frame(
    read.csv("DataSet/uk-2007-05.obvious_features.csv", header = T)
  )

  # Join both datasets by Host ID
  dataset <- merge(raw_labels, raw_features, by.x = "V1", by.y = "X.hostid")

  dataset <-
    subset(
      dataset,
      V3 != "-",
      select = c("V1", "V2", "number_of_pages", "length_of_hostname", "V3")
    )

  # Convert the labels to numeric
  dataset$V3 <-
    as.numeric(as.character(dataset$V3))

  dataset$binned_y <-
    as.factor(
      if (round) {
        # Bin the spam/not spam into either 1 or 0
        sapply(
          dataset$V3,
          (function(x) if (x >= 0.75) "spam" else "nonspam")
        )
      } else {
        # Use the original spam/nonspam/undecided labels
        as.factor(dataset$V2)
      }
    )

  # Select the columns
  dataset <- dataset[, c("number_of_pages", "length_of_hostname", "binned_y")]

  # Scale with log10
  dataset$number_of_pages <- log10(dataset$number_of_pages)
  dataset$length_of_hostname <- log10(dataset$length_of_hostname)

  # Grab a list of spam rows, and sample the same number of non-spam rows
  dataset_spam <- subset(dataset, binned_y == "spam")
  dataset_not_spam <-
    dataset[
      sample(
        nrow(dataset),
        size = nrow(dataset_spam),
        replace = F
      ),
    ]

  # Combine the spam and non-spam rows
  bound <- rbind(dataset_spam, dataset_not_spam)

  # Return the dataset
  return(bound)
}

data_train <- prepare_dataset(
  "DataSet/webspam-uk2007-set1-1-10_TRAINING/WEBSPAM-UK2007-SET1-labels.txt",
  round = T
)

data_test <- prepare_dataset(
  "DataSet/webspam-uk2007-set2-1-10_TEST/WEBSPAM-UK2007-SET2-labels.txt",
  round = T
)

str(data_train)
summary(data_train)

# Check out the dataset with a box plot
boxplot(length_of_hostname ~ number_of_pages, data = data_train)
boxplot(number_of_pages ~ length_of_hostname, data = data_train)

boxplot(length_of_hostname ~ binned_y, data = data_train)
boxplot(number_of_pages ~ binned_y, data = data_train)

# count the number of rows where number of pages > 1000
nrow(data_train[data_train$number_of_pages > 1000, ])
nrow(data_train[data_train$number_of_pages <= 1000, ])

# Box plot for the number of pages <= 1000
boxplot(
  number_of_pages ~ binned_y,
  data = data_train[data_train$number_of_pages <= 1000, ]
)

# X,Y plot graphed with ggplot2
View(
  ggplot(
    data_train,
    aes(
      y = number_of_pages,
      x = length_of_hostname,
      shape = as.character(binned_y),
      color = as.character(binned_y),
    )
  ) +
    geom_point() +
    # scale_y_continuous(trans = "log10") +
    ggtitle("Plotted \"Obvious Features\"")
)

# Use SVM to classify the data
summary(
  radial_svm <- svm(
    binned_y ~ .,
    data = data_train,
    kernel = "radial",
    gamma = 0.1,
    cost = 1,
    cross = 10,
  )
)

# Plot the decision boundary
View(plot(radial_svm, data = data_train))

# Print the confusion matrix
table(data_test$binned_y, predict(radial_svm, data_test))

#           nonspam spam
#   nonspam      80   29
#   spam         53   66

# Print the accuracy
accuracy <- function(svm, data) {
  return(
    sum(
      predict(svm, data) == data$binned_y
    ) / nrow(data)
  )
}

accuracy(radial_svm, data_train)
# [1] 0.6390244

accuracy(radial_svm, data_test)
# [1] 0.6403509

# Automated tuning
tune_result <-
  tune.svm(
    binned_y ~ .,
    data = data_train,
    kernel = "radial",
    cross = 10,
    gamma = seq(0.01, 0.1, by = 0.05),
    cost = seq(0.01, 0.1, by = 0.05)
  )

summary(tune_result)

# Parameter tuning of ‘svm’:
#
# - sampling method: 10-fold cross validation
#
# - best parameters:
#  gamma cost
#   0.01 0.01
#
# - best performance: 0.4756098
#
# - Detailed performance results:
#   gamma cost     error dispersion
# 1  0.01 0.01 0.4756098 0.06922515
# 2  0.06 0.01 0.4756098 0.06922515
# 3  0.01 0.06 0.4756098 0.06922515
# 4  0.06 0.06 0.4756098 0.06922515