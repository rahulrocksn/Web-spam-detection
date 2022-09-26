# Random Forest Testing

library(mltools)
library(data.table)
library(randomForest)

set.seed(10)

# Supporting Function 

binningSpam <- function(value){
  if (value == 0.5 & F){
    return(factor("Undecided"))
  }
  if (value> 0.9){
    return(factor("Spam"))
  }
  else{
    return(factor("NonSpam"))
  }
}

# RAW FILES

trainRAW <- data.frame(read.csv("webspam-uk2007-set1-1-10_TRAINING/WEBSPAM-UK2007-SET1-labels.txt", header= F, sep=" ", dec="."))
testRAW <- data.frame(read.csv("webspam-uk2007-set2-1-10_TEST/WEBSPAM-UK2007-SET2-labels.txt", header= F, sep=" ", dec="."))
linkTransfromedRAW <- data.frame(read.csv("uk-2007-05.link_based_features_transformed.csv", header=T))
linkContentBasedRAW <- data.frame(read.csv("uk-2007-05.content_based_features.csv", header=T))

# Training Data

linkTransformedTrain <- merge(trainRAW,linkTransfromedRAW,by.x = "V1", by.y="X.hostid")
linkTransformedTrain <- merge(linkTransformedTrain, linkContentBasedRAW, by.x = "V1", by.y="X.hostid")
linkTransformedTrain <- linkTransformedTrain[,c(-1,-2,-4)]
linkTransformedTrain$V3 <- as.numeric(as.character(linkTransformedTrain$V3))
linkTransformedTrain <- na.omit(linkTransformedTrain)
linkTransformedTrain$binnedY <- sapply(linkTransformedTrain$V3, binningSpam)

# Testing Data

testingData <- merge(testRAW,linkTransfromedRAW,by.x = "V1", by.y="X.hostid")
testingData <- merge(testingData, linkContentBasedRAW, by.x = "V1", by.y="X.hostid")
testingData <- testingData[,c(-1,-2,-4)]
testingData$V3 <- as.numeric(as.character(testingData$V3))
testingData <- na.omit(testingData)
testingData$binnedY <- sapply(testingData$V3, binningSpam)

{
  linkTransformedTrainSpam <- subset(linkTransformedTrain, V3 > 0.5)
  linkTransformedTrainSpam <- rbind(linkTransformedTrainSpam,linkTransformedTrainSpam) # *2 spam entries
  linkTransformedTrainNotSpam <- subset(linkTransformedTrain, V3 < 0.5)
  linkTransformedTrainNotSpam <- linkTransformedTrainNotSpam[sample(1:nrow(linkTransformedTrainNotSpam), size=nrow(linkTransformedTrainSpam), replace=F),]
  
  linkTransformedBalanced = rbind(linkTransformedTrainSpam, linkTransformedTrainNotSpam)
}


# For cor
linkTransformedBalancedNoFactor <- subset(linkTransformedTrain, select=-c(binnedY,hostname))

corTable2 <- abs(cor(linkTransformedBalancedNoFactor, y=linkTransformedTrain$V3))
corTable2 = corTable2[order(corTable2, decreasing = T),,drop=F]
headhead <- head(corTable2,21)
headhead

# RFCV test
linkTransformedBalancedY <- linkTransformedBalanced$binnedY
linkTransformedBalancedX <- subset(linkTransformedBalanced,select=-c(binnedY,V3))

result <- rfcv(linkTransformedBalancedX,linkTransformedBalancedY, recursive = T)
with(result, plot(n.var, error.cv, log="x", type="o", lwd=2))


forest.All <- randomForest(binnedY ~ . -V3 -hostname, data = linkTransformedBalanced, nodesize=1, ntree=500, mtry=13)
plot(forest.All)

forestPredict.All <- predict(forest.All, testingData, type="class")
tempTable.All <- table(testingData$binnedY, forestPredict.All)
tempTable.All

# forestPredict.All
#         NonSpam Spam
# NonSpam    1597  289
# Spam         43   70

