# Random Forest Testing

library(mltools)
library(data.table)
library(randomForest)
library(ROCR)

set.seed(10)

# Dataset ---------------------------------------------------------------------

{
  raw_train <- data.frame(read.csv(
    "DataSet/webspam-uk2007-set1-1-10_TRAINING/WEBSPAM-UK2007-SET1-labels.txt",
    header = F, sep = " ", dec = "."
  ))
  raw_test <- data.frame(read.csv(
    "DataSet/webspam-uk2007-set2-1-10_TEST/WEBSPAM-UK2007-SET2-labels.txt",
    header = F, sep = " ", dec = "."
  ))
  raw_link_transformed <- data.frame(read.csv(
    "DataSet/uk-2007-05.link_based_features_transformed.csv",
    header = T
  ))
  raw_content_based <- data.frame(read.csv(
    "DataSet/uk-2007-05.content_based_features.csv",
    header = T
  ))
}

prepare_raw <- function(raw) {
  # Join both datasets by Host ID
  dataset <- merge(raw, raw_link_transformed, by.x = "V1", by.y = "X.hostid")
  dataset <- merge(dataset, raw_content_based, by.x = "V1", by.y = "X.hostid")

  # Remove columns
  # - V1 (HostID)
  # - V2 (spam/nonspam labels)
  # - V4 (raw results)
  # - hostname
  dataset <- subset(dataset, select = -c(V1, V2, V4, hostname))

  # Rename column V3 (spamicity)
  names(dataset)[1] <- "spamicity"

  # Convert spamicity to numeric
  dataset$spamicity <- as.numeric(dataset$spamicity)

  # Drop NAs
  dataset <- na.omit(dataset)

  # Bin spam/nonspam
  dataset$binned_y <- sapply(dataset$spamicity, function(value) {
    if (value > 0.5) {
      return(factor("Spam"))
    } else {
      return(factor("NonSpam"))
    }
  })

  return(dataset)
}

set_train <- prepare_raw(raw_train)
set_test <- prepare_raw(raw_test)
View(set_train)

# Principal Component Analysis ------------------------------------------------

prepare_pca_x <- function(dataset) {
  return(subset(dataset, select = -c(spamicity, binned_y)))
}

frame_pca <- function(dataset, pca_x, ncols) {
  set_pca <- data.frame(dataset$binned_y, pca_x)[, c(1:ncols)]
  names(set_pca)[1] <- "binned_y"
  return(set_pca)
}

# Perform PCA on training data
# Select all columns except the target (Y)
set_train_x <- prepare_pca_x(set_train)
set_train_pca_params <- prcomp(set_train_x, center = TRUE, scale = TRUE)
summary(set_train_pca_params$rotation)

# Create a new training dataframe containing the Y and the first 9 PCA params
set_train_pca_x <- set_train_pca_params$x
set_train_pca <- frame_pca(set_train, set_train_pca_x, 10)
head(set_train_pca)

# Transform the test data using the training set's PCA
set_test_x <- prepare_pca_x(set_test)
set_test_pca_x <- scale(set_test_x) %*% set_train_pca_params$rotation

# Create a new testing dataframe containing the Y and the first 9 PCA params
set_test_pca <- frame_pca(set_test, set_test_pca_x, 10)
head(set_test_pca)

# Balancing the dataset -------------------------------------------------------

sample_balanced <- function(dataset) {
  ds_spam <- subset(dataset, binned_y == "Spam")
  ds_nonspam <- subset(dataset, binned_y == "NonSpam")
  n_samp <- min(nrow(ds_spam), nrow(ds_nonspam))

  return(rbind(
    ds_spam[sample(seq(1, nrow(ds_spam)), size = n_samp), ],
    ds_nonspam[sample(seq(1, nrow(ds_nonspam)), size = n_samp), ]
  ))
}

set_train_pca_bal <- sample_balanced(set_train_pca)
set_test_pca_bal <- sample_balanced(set_test_pca)

# Random Forest Cross Validation ----------------------------------------------

result <-
  rfcv(set_train_pca_bal[, -1], set_train_pca_bal$binned_y, recursive = TRUE)
with(result, plot(n.var, error.cv, log = "x", type = "o", lwd = 2))

# Random Forest ---------------------------------------------------------------

forest <- randomForest(
  binned_y ~ .,
  data = set_train_pca_bal,
  ntree = 500,
  nodesize = 1
)
plot(forest)

measure_perf <- function(dataset) {
  forest_predict <- predict(forest, dataset, type = "class")
  print(table(dataset$binned_y, forest_predict))
  #          forest_predict
  #           NonSpam Spam
  #   NonSpam      83   30
  #   Spam         29   84

  print(sum(dataset$binned_y == forest_predict) / length(forest_predict))
  # [1] 0.7389381

  # Plot ROC curve
  perf <- performance(
    prediction(
      as.numeric(forest_predict),
      as.numeric(dataset$binned_y)
    ),
    "tpr", "fpr"
  )
  View(plot(perf))
}

measure_perf(set_train_pca_bal)
measure_perf(set_test_pca_bal)

# ----------------------------------------------------------------------------

if (FALSE) {
  {
    linkTransformedTrainSpam <- subset(linkTransformedTrain, V3 > 0.5)
    linkTransformedTrainSpam <- rbind(linkTransformedTrainSpam, linkTransformedTrainSpam) # *2 spam entries
    linkTransformedTrainNotSpam <- subset(linkTransformedTrain, V3 < 0.5)
    linkTransformedTrainNotSpam <- linkTransformedTrainNotSpam[sample(1:nrow(linkTransformedTrainNotSpam), size = nrow(linkTransformedTrainSpam), replace = F), ]

    linkTransformedBalanced <- rbind(linkTransformedTrainSpam, linkTransformedTrainNotSpam)
  }


  # For cor
  linkTransformedBalancedNoFactor <- subset(linkTransformedTrain, select = -c(binnedY, hostname))

  corTable2 <- abs(cor(linkTransformedBalancedNoFactor, y = linkTransformedTrain$V3))
  corTable2 <- corTable2[order(corTable2, decreasing = T), , drop = F]
  headhead <- head(corTable2, 21)
  headhead

  # RFCV test
  linkTransformedBalancedY <- linkTransformedBalanced$binnedY
  linkTransformedBalancedX <- subset(linkTransformedBalanced, select = -c(binnedY, V3))

  result <- rfcv(linkTransformedBalancedX, linkTransformedBalancedY, recursive = T)
  with(result, plot(n.var, error.cv, log = "x", type = "o", lwd = 2))


  forest.All <- randomForest(binnedY ~ . - V3 - hostname, data = linkTransformedBalanced, nodesize = 1, ntree = 500, mtry = 13)
  plot(forest.All)

  forestPredict.All <- predict(forest.All, testingData, type = "class")
  tempTable.All <- table(testingData$binnedY, forestPredict.All)
  tempTable.All

  # forestPredict.All
  #         NonSpam Spam
  # NonSpam    1597  289
  # Spam         43   70
}