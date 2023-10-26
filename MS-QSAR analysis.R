library(dplyr)
library(e1071)
library(randomForest)
library(proxy)
library(mltools) 

library(readr)
library(ISLR2)
library(class)
library(caret)
library(ggplot2)

msqsar_cl <- function(x, y, p = 0.8, 
                      min_size = 3, max_size = 7, max_dist = 1, seed = 1,
                      model_type = "svm", k = "radial", cost = 1, ntrees = 500) {
    set.seed(seed)
    
    # Split x and y into training and test
    index <- sort(sample(nrow(x), nrow(x) * p, replace = FALSE), decreasing = F)
    x.train <- x[index,]
    y.train <- y[index]
    x.test <- x[-index,]
    y.test <- y[-index]
    
    # Create training matrix and test matrix
    train_dist_matrix <- as.matrix(dist(x.train))
    test_dist_matrix <- as.data.frame.matrix(dist(x.train, x.test))
    
    # Calculate the mean distance
    mean_dist <- mean(train_dist_matrix)
    
    # Calculate the predictions
    pred_values <- sapply(1:ncol(test_dist_matrix), function(i) {
      
      # Wrapping the code in tryCatch to handle exceptions
      result <- tryCatch({
        
        # Creating a data frame with distances and original indices
        distance_matrix <- data.frame(distances = test_dist_matrix[, i], 
                                      original_index = rownames(test_dist_matrix))
        
        # Calculating the number of elements to keep within the specified range
        num_elements_to_keep <- min(max_size, nrow(distance_matrix))
        
        # If the number of elements to keep is less than the minimum size, return NA
        if (num_elements_to_keep < min_size) {
          return(NA_character_)
        }
        
        # Keep the smallest elements within the range
        # and filter the selected observations based on max_dist
        filtered_df <- distance_matrix %>%
          top_n(-num_elements_to_keep, wt = distances) %>%
          arrange(distances) %>%
          slice_head(n = num_elements_to_keep) %>%
          filter(distances <= max_dist*mean_dist)
        
        # If after filtering by max_dist, 
        # the number of observations is less than min_size, return NA
        if (nrow(filtered_df) < min_size) {
          return(NA_character_)
        }
        
        # If filtered_df has only one unique species, return that species
        unique_species <- unique(y[as.integer(filtered_df$original_index)])
        if (length(unique_species) == 1) {
          return(as.character(unique_species))
        } else {
          
          # For cases where there are fewer classes than original but more than one, 
          # proceed to build the model with the available classes
          
          # Get the indices to keep
          indices <- as.integer(filtered_df$original_index)
          
          # Create the test matrix
          test_matrix <- data.frame(
            y.test[i], 
            data.frame(
              t(test_dist_matrix[rownames(test_dist_matrix) %in% indices, ]))[i,])
          
          # Create the train matrix
          train_matrix <- data.frame(
            y[indices], 
            train_dist_matrix[rownames(train_dist_matrix) %in% indices, 
                              colnames(train_dist_matrix) %in% indices])
          colnames(train_matrix)[1] <- "y.train"
          
          # Choose model type based on model_type argument
          if (model_type == "svm") {
            model <- svm(y.train ~ ., train_matrix, 
                         kernel = k, cost = cost, type = "C-classification")
          } else if (model_type == "rf") {
            train_matrix$y.train <- factor(train_matrix$y.train)
            model <- randomForest(train_matrix$y.train ~ ., train_matrix, ntree = ntrees)
          } else {
            stop("Invalid model_type. Please choose 'svm' or 'rf'.")
          }
          
          return(as.character(predict(model, newdata = test_matrix)))
        }
        
      }, error = function(e) {
        # Handle errors and return NA
        warning("Error occurred for column ", i, ": ", conditionMessage(e))
        return(NA_character_)
      })
    })
    
    # Calculate the percentage of predictions done
    perc_pred_values <- length(
      pred_values[!is.na(pred_values)]) / length(pred_values) * 100
    
    # Calculate Matthew's Correlation Coefficient
    MCC <- mcc(pred_values, as.character(y.test))
    
    # Return the predictions, the percentage and the MCC
    return(list("pred values" = pred_values, 
                "percentage of predicted values" = perc_pred_values,
                "MCC" = MCC))
}


msqsar_reg <- function(x, y, p = 0.8,
                       min_size = 3, max_size = 7, max_dist = 4, seed = 1,
                       model_type = "svm", k = "radial", cost = 1, ntrees = 500) {
  suppressWarnings({
    set.seed(seed)
    
    # Split x and y into training and test
    index <- sort(sample(nrow(x), nrow(x) * 0.8, replace = FALSE), decreasing = F)
    x.train <- x[index,]
    y.train <- y[index]
    x.test <- x[-index,]
    y.test <- y[-index]
    
    # Create training matrix and test matrix
    train_dist_matrix <- as.matrix(dist(x.train))
    test_dist_matrix <- as.data.frame.matrix(dist(x.train, x.test))
    
    # Calculate the mean distance
    mean_dist <- mean(train_dist_matrix)
    
    # Calculate the predictions
    pred_values <- sapply(1:ncol(test_dist_matrix), function(i) {
      # Wrapping the code in tryCatch to handle exceptions
      result <- tryCatch({
        # Creating a data frame with distances and original indices
        distance_matrix <- data.frame(distances = test_dist_matrix[, i],
                                      original_index = rownames(test_dist_matrix))
        
        # Calculating the number of elements to keep within the specified range
        num_elements_to_keep <- min(max_size, nrow(distance_matrix))
        
        # If the number of elements to keep is less than the minimum size, return NA
        if (num_elements_to_keep < min_size) {
          return(NA)
        }
        
        # Keep the smallest elements within the range
        # and filter the selected observations based on max_dist
        filtered_df <- distance_matrix %>%
          top_n(-num_elements_to_keep, wt = distances) %>%
          arrange(distances) %>%
          slice_head(n = num_elements_to_keep) %>% 
          filter(distances <= max_dist*mean_dist)
        
        # If after filtering by max_dist, 
        # the number of observations is less than min_size, return NA
        if (nrow(filtered_df) < min_size) {
          return(NA)
        }
        
        # Get the indices to keep
        indices <- as.integer(filtered_df$original_index)
        
        # Create the test matrix
        test_matrix <- data.frame(
          y.test[i], 
          data.frame(
            t(test_dist_matrix[rownames(test_dist_matrix) %in% indices, ]))[i,])
        
        # Create the train matrix
        train_matrix <- data.frame(
          y[indices], 
          train_dist_matrix[rownames(train_dist_matrix) %in% indices, 
                            colnames(train_dist_matrix) %in% indices])
        colnames(train_matrix)[1] <- "y.train"
        
        # Choose model type based on model_type argument
        if (model_type == "svm") {
          model <- svm(y.train ~ ., train_matrix, 
                       kernel = k, cost = cost, type = "eps-regression")
        } else if (model_type == "rf") {
          model <- randomForest(y.train ~ ., train_matrix, ntree = ntrees)
        } else {
          stop("Invalid model_type. Please choose 'svm' or 'rf'.")
        }
        
        return(predict(model, newdata = test_matrix))
        
      }, error = function(e) {
        # Handle errors and return NA
        warning("Error occurred for column ", i, ": ", conditionMessage(e))
        return(NA)
      })
    })
    
    # Calculate the percentage of predictions done
    perc_pred_values <- length(
      pred_values[!is.na(pred_values)]) / length(pred_values) * 100
    
    # Calculate the Mean Square Error
    mse <- mean((pred_values - y.test)^2, na.rm = T)
    
    # Return the predictions, the percentage and the MSE
    return(list("pred values" = round(pred_values,3),
                "percentage of predicted values" = perc_pred_values, 
                "MSE" = mse))
  })
}


diabetes <- read_csv("diabetes.csv")
df <- diabetes[1:60,]

y <- as.factor(df$Outcome)
x <- data.frame(df[,-ncol(df)])
x <- x[,c(1,2)]

predi_values <- msqsar_cl(x, y, min_size = 3, max_size = 5, max_dist = 1, seed = 1, model_type = "svm", k = "radial", cost = 1)
predi_values

predi_values <- msqsar_cl(x, y, min_size = 3, max_size = 5, max_dist = 1, seed = 1, model_type = "rf")
predi_values

{
num_iterations <- 100
best_MCC <- -Inf
best_params <- list()

for (i in 1:num_iterations) {
  # Randomly select parameters within a defined range
  min_val <- sample(2:5, 1)
  max_val <- sample(5:10, 1)
  dist_val <- runif(1, 0.1, 1)
  
  pred_values <- msqsar_cl(x, y, min_size = min_val, max_size = max_val, max_dist = dist_val, seed = 1, model_type = "svm", k = "linear", cost = 1)
  
  if (pred_values$MCC > best_MCC) {
    best_MCC <- pred_values$MCC
    best_params$min_size <- min_val
    best_params$max_size <- max_val
    best_params$max_dist <- dist_val
  }
}

print(best_params)
print(best_MCC)
}


# Diabetes ----

df <- diabetes

y <- as.factor(df$Outcome)
x <- data.frame(df[,-ncol(df)])


set.seed(1)
index <- sort(sample(nrow(df), nrow(df) * 0.8, replace = FALSE), decreasing = F)
traindf <- df[index,]
testdf <- df[-index,]

## Tuning ----
param_grid <- expand.grid(min_size = c(3, 4, 5),
                          max_size = c(6, 8, 10),
                          max_dist = c(0.5, 1, 1.5))



results <- apply(param_grid, 1, function(params) {
  min_val <- params["min_size"]
  max_val <- params["max_size"]
  dist_val <- params["max_dist"]
  
  pred_values <- msqsar_cl(x, y, min_size = min_val, max_size = max_val, max_dist = dist_val, seed = 1, model_type = "rf", k = "radial", cost = 1)
  
  return(pred_values$MCC)
})

param_grid$MCC <- results

best_parameters <- param_grid[which.max(param_grid$MCC),]
best_parameters

## MSQSAR -----
predi_values <- msqsar_cl(x, y, min_size = 3, max_size = 10, max_dist = 1, seed = 1, model_type = "svm", k = "radial", cost = 1)
predi_values

## log regression ----
model <- glm(as.factor(traindf$Outcome) ~. ,data= traindf ,family="binomial")
p.hat<- predict(model ,newdata=testdf,type="response")
preds <- ifelse(p.hat> 0.5, 0, 1)
MCC <- mcc(preds, testdf$Outcome)
MCC 

## SVM ----
model <- svm(traindf$Outcome ~ ., traindf, kernel = "radial",
             cost = 1, type = "C-classification")

preds <- predict(model, newdata = testdf)

MCC <- mcc(preds, as.factor(testdf$Outcome))
MCC 


## RF ----
predi_values <- msqsar_cl(x, y, min_size = 3, max_size = 10, max_dist = 1, seed = 1, model_type = "rf", ntrees = 500)
predi_values


model <- randomForest(as.factor(traindf$Outcome) ~ ., traindf)

preds <- predict(model, newdata = testdf)

MCC <- mcc(preds, as.factor(testdf$Outcome))
MCC 

## KNN ----

best_k = 3
set.seed(1)
preds <- knn(train = traindf[,-ncol(traindf)],test= testdf[,-ncol(testdf)],
             cl = traindf$Outcome, k=best_k)

MCC <- mcc(preds, as.factor(testdf$Outcome))
MCC

# Breast Cancer ----

BreastCancer <- read_csv("data.csv")

df <- data.frame(BreastCancer[,-c(1,33)])

y <- as.factor(df$diagnosis)
x <- data.frame(df[,-1])

set.seed(1)
index <- sort(sample(nrow(df), nrow(df) * 0.8, replace = FALSE), decreasing = F)
traindf <- df[index,]
testdf <- df[-index,]
## Tuning ----
param_grid <- expand.grid(min_size = c(3, 4, 5),
                          max_size = c(6, 8, 10),
                          max_dist = c(0.5, 1, 1.5))



results <- apply(param_grid, 1, function(params) {
  min_val <- params["min_size"]
  max_val <- params["max_size"]
  dist_val <- params["max_dist"]
  
  pred_values <- msqsar_cl(x, y, min_size = min_val, max_size = max_val, max_dist = dist_val, seed = 1, model_type = "svm", k = "radial", cost = 1)
  
  return(pred_values$MCC)
})

param_grid$MCC <- results

best_parameters <- param_grid[which.max(param_grid$MCC),]
best_parameters

## MSQSAR ----
predi_values <- msqsar_cl(x, y, min_size = 3, max_size = 8, max_dist = 1.5, seed = 1, model_type = "svm", k = "radial", cost = 1)

predi_values$MCC

## log regression ----
model <- glm(as.factor(traindf$diagnosis) ~. ,data= traindf ,family="binomial")
p.hat<-predict(model ,newdata=testdf,type="response")
preds <- ifelse(p.hat> 0.5,"M","B")
MCC <- mcc(preds, testdf$diagnosis)
MCC 

## SVM ----
model <- svm(as.factor(traindf$diagnosis) ~ ., traindf, kernel = "radial",
             cost = 1, type = "C-classification")

preds <- predict(model, newdata = testdf)

MCC <- mcc(preds, as.factor(testdf$diagnosis))
MCC 

## RF ----
predi_values <- msqsar_cl(x, y, min_size = 3, max_size = 8, max_dist = 1.5, seed = 1, model_type = "rf", ntrees = 500)
predi_values


model <- randomForest(as.factor(traindf$diagnosis) ~ ., traindf)

preds <- predict(model, newdata = testdf)

MCC <- mcc(preds, as.factor(testdf$diagnosis))
MCC #

## KNN ----

best_k = 3
set.seed(1)
preds <- knn(train = traindf[,-1],test= testdf[,-1],
             cl = traindf$diagnosis, k=best_k)

MCC <- mcc(preds, as.factor(testdf$diagnosis))
MCC

# Covertype ----

df1 <- read_delim(gzfile("covtype.data.gz"), delim = ",", col_names = FALSE)
x <- data.frame(df1[1:500, 1:10])
y <- as.factor(df1$X55[1:500])
head(x)


set.seed(1)
index <- sort(sample(nrow(x), nrow(x) * 0.8, replace = FALSE), decreasing = F)
x.train <- x[index,]
y.train <- y[index]

x.test <- x[-index,]
y.test <- y[-index]

testdf <- data.frame(y.test, x.test)
traindf <- data.frame(y.train, x.train)
## Tuning ----
param_grid <- expand.grid(min_size = c(3, 4, 5),
                          max_size = c(6, 8, 10),
                          max_dist = c(0.5, 1, 1.5))



results <- apply(param_grid, 1, function(params) {
  min_val <- params["min_size"]
  max_val <- params["max_size"]
  dist_val <- params["max_dist"]
  
  pred_values <- msqsar_cl(x, y, min_size = min_val, max_size = max_val, max_dist = dist_val, seed = 1, model_type = "svm", k = "radial", cost = 1)
  
  return(pred_values$MCC)
})

param_grid$MCC <- results

best_parameters <- param_grid[which.max(param_grid$MCC),]
best_parameters

## MSQSAR ----
predi_values <- msqsar_cl(x, y, min_size = 3, max_size = 6, max_dist = 0.5, seed = 1, model_type = "svm", k = "radial", cost = 1)
predi_values


## log regression ----
model <- glm(as.factor(traindf$y.train) ~. ,data= traindf ,family="binomial")
p.hat<- predict(model ,newdata=testdf,type="response")
preds <- ifelse(p.hat> 0.5, 0, 1)
MCC <- mcc(as.factor(preds), testdf$y.test)
MCC 

## SVM ----
set.seed(1)

model <- svm(traindf$y.train ~ ., traindf, kernel = "radial",
             cost = 1, type = "C-classification")

preds <- predict(model, newdata = testdf)

MCC <- mcc(preds, testdf$y.test)
MCC 


## RF ----
model <- randomForest(as.factor(traindf$y.train) ~ ., traindf)

preds <- predict(model, newdata = testdf)

MCC <- mcc(preds, testdf$y.test)
MCC 

## KNN ----
library(class)
best_k = 3
set.seed(1)
preds <- knn(train = traindf[,-1],test= testdf[,-1],
             cl = traindf$y.train, k=best_k)

MCC <- mcc(preds, as.factor(testdf$y.test))
MCC









# Auto  ----
df <- Auto[,-c(8,9)]

y <- df$mpg
x <- data.frame(df[,-1])

set.seed(1)
index <- sort(sample(nrow(df), nrow(df) * 0.8, replace = FALSE), decreasing = F)
traindf <- df[index,]
testdf <- df[-index,]
## Tuning ----
param_grid <- expand.grid(min_size = c(3, 4, 5),
                          max_size = c(6, 8, 10),
                          max_dist = c(0.5, 1, 1.5))



results <- apply(param_grid, 1, function(params) {
  min_val <- params["min_size"]
  max_val <- params["max_size"]
  dist_val <- params["max_dist"]
  
  pred_values <- msqsar_reg(x, y, min_size = min_val, max_size = max_val, max_dist = dist_val, seed = 1, model_type = "svm", k = "radial", cost = 1)
  
  return(pred_values$MSE)
})

param_grid$MSE <- results

best_parameters <- param_grid[which.min(param_grid$MSE),]
best_parameters

## MSQSAR ----
predi_values <- msqsar_reg(x, y, min_size = 3, max_size = 8, max_dist = 0.5, seed = 1, model_type = "svm", k = "radial", cost = 1)
predi_values$MSE


## log regression ----
model <- lm(traindf$mpg ~. ,data= traindf)
preds <-predict(model ,newdata=testdf)

MSE <- mean((preds - testdf$mpg)^2)
MSE 

## SVM ----


model <- svm(traindf$mpg ~ ., traindf, kernel = "radial",
             cost = 1, type = "eps-regression")

preds <- predict(model, newdata = testdf)

MSE <- mean((preds - testdf$mpg)^2)
MSE 

## RF ----
model <- randomForest(traindf$mpg ~ ., traindf)

preds <- predict(model, newdata = testdf)

MSE <- mean((preds - testdf$mpg)^2)
MSE 


## KNN ----
best_k = 3
set.seed(1)

knn_model <- train(mpg ~ ., data = traindf, method = "knn", tuneGrid = data.frame(k = best_k))
preds <- predict(knn_model, testdf)
MSE <- mean((preds - testdf$mpg)^2)
MSE



# Boston ----
df <- Boston

y <- df$medv
x <- data.frame(df[,-13])

set.seed(1)
index <- sort(sample(nrow(df), nrow(df) * 0.8, replace = FALSE), decreasing = F)
traindf <- df[index,]
testdf <- df[-index,]
## Tuning ----
param_grid <- expand.grid(min_size = c(3, 4, 5),
                          max_size = c(6, 8, 10),
                          max_dist = c(0.5, 1, 1.5))



results <- apply(param_grid, 1, function(params) {
  min_val <- params["min_size"]
  max_val <- params["max_size"]
  dist_val <- params["max_dist"]
  
  pred_values <- msqsar_reg(x, y, min_size = min_val, max_size = max_val, max_dist = dist_val, seed = 1, model_type = "svm", k = "radial", cost = 1)
  
  return(pred_values$MSE)
})

param_grid$MSE <- results

best_parameters <- param_grid[which.min(param_grid$MSE),]
best_parameters

## MSQSAR ----
predi_values <- msqsar_reg(x, y, min_size = 3, max_size = 6, max_dist = 0.5, seed = 1, model_type = "svm", k = "radial", cost = 1)
predi_values$MSE


## log regression ----
model <- lm(traindf$medv ~. ,data= traindf)
preds <-predict(model ,newdata=testdf)

MSE <- mean((preds - testdf$medv)^2)
MSE 

## SVM ----

model <- svm(traindf$medv ~ ., traindf, kernel = "radial",
             cost = 1, type = "eps-regression")

preds <- predict(model, newdata = testdf)

MSE <- mean((preds - testdf$medv)^2)
MSE 

## RF ----
model <- randomForest(traindf$medv ~ ., traindf)

preds <- predict(model, newdata = testdf)

MSE <- mean((preds - testdf$medv)^2)
MSE 


## KNN ----
best_k = 3
set.seed(1)

knn_model <- train(medv ~ ., data = traindf, method = "knn", tuneGrid = data.frame(k = best_k))
preds <- predict(knn_model, testdf)
MSE <- mean((preds - testdf$medv)^2)
MSE



# Forest Fires ----
forestfires <- read_csv("forestfires.csv")

df <- forestfires[,-c(3,4)]

y <- df$area
x <- data.frame(df[,-11])

set.seed(1)
index <- sort(sample(nrow(df), nrow(df) * 0.8, replace = FALSE), decreasing = F)
traindf <- df[index,]
testdf <- df[-index,]
## Tuning ----
param_grid <- expand.grid(min_size = c(3, 4, 5),
                          max_size = c(6, 8, 10),
                          max_dist = c(0.5, 1, 1.5))



results <- apply(param_grid, 1, function(params) {
  min_val <- params["min_size"]
  max_val <- params["max_size"]
  dist_val <- params["max_dist"]
  
  pred_values <- msqsar_reg(x, y, min_size = min_val, max_size = max_val, max_dist = dist_val, seed = 1, model_type = "svm", k = "radial", cost = 1)
  
  return(pred_values$MSE)
})

param_grid$MSE <- results

best_parameters <- param_grid[which.min(param_grid$MSE),]
best_parameters

## MSQSAR ----
predi_values <- msqsar_reg(x, y, min_size = 3, max_size = 6, max_dist = 0.5, seed = 1, model_type = "svm", k = "radial", cost = 1)
predi_values$MSE


## log regression ----
model <- lm(traindf$area ~. ,data= traindf)
preds <-predict(model ,newdata=testdf)

MSE <- mean((preds - testdf$area)^2)
MSE 

## SVM ----

model <- svm(traindf$area ~ ., traindf, kernel = "radial",
             cost = 1, type = "eps-regression")

preds <- predict(model, newdata = testdf)

MSE <- mean((preds - testdf$area)^2)
MSE 

## RF ----

model <- randomForest(traindf$area ~ ., traindf)

preds <- predict(model, newdata = testdf)

MSE <- mean((preds - testdf$area)^2)
MSE 

## KNN ----
best_k = 3
set.seed(1)

knn_model <- train(area ~ ., data = traindf, method = "knn", tuneGrid = data.frame(k = best_k))
preds <- predict(knn_model, testdf)
MSE <- mean((preds - testdf$area)^2)
MSE



# System time ----
x <- data.frame(df1[1:1500, 1:10])
y <- as.factor(df1$X55[1:1500])

set.seed(1)
index <- sort(sample(nrow(x), nrow(x) * 0.8, replace = FALSE), decreasing = F)
x.train <- x[index,]
y.train <- y[index]

x.test <- x[-index,]
y.test <- y[-index]

testdf <- data.frame(y.test, x.test)
traindf <- data.frame(y.train, x.train)

system.time({
  model <- svm(y.train ~ ., traindf, kernel = "radial",
               cost = 1, type = "C-classification")
  
  preds <- predict(model, newdata = testdf)
  
  MCC <- mcc(preds, as.factor(y.test))
})



system.time({
  predi_values <- msqsar_cl(x, y, min_size = 3, max_size = 6, max_dist = 0.5, seed = 1, model_type = "svm", k = "radial", cost = 1)
})


observations <- c(500, 1000, 1500)
user_time <- c(2.154, 4.611, 7.092)  # Replace with your actual user time values
system_time <- c(0.095, 0.181, 0.343)  # Replace with your actual system time values
elapsed_time <- c(2.270, 4.835, 7.561)  # Replace with your actual elapsed time values

user_time <- c(0.020, 0.046, 0.080)  # Replace with your actual user time values
system_time <- c(0.002, 0.003, 0.003)  # Replace with your actual system time values
elapsed_time <- c(0.027, 0.051, 0.083) 

# Create a data frame
data <- data.frame(Observations = observations, User = user_time, System = system_time, Elapsed = elapsed_time)


# Create a line chart using ggplot2
ggplot(data, aes(x = Observations)) +
  geom_line(aes(y = User, color = "User")) +
  geom_line(aes(y = System, color = "System")) +
  geom_line(aes(y = Elapsed, color = "Elapsed")) +
  geom_point(data = data, aes(y = User), color = "red", size = 3, shape = 19) +
  geom_point(data = data, aes(y = System), color = "blue", size = 3, shape = 19) +
  geom_point(data = data, aes(y = Elapsed), color = "green", size = 3, shape = 19) +
  labs(x = "Number of Observations", y = "Time", title = "System Time vs. Number of Observations") +
  scale_color_manual(values = c("User" = "red", "System" = "blue", "Elapsed" = "green")) +
  theme_minimal() +
  theme(legend.title = element_blank())






