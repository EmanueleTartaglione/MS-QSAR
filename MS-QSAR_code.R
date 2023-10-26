library(dplyr)
library(e1071)
library(randomForest)
library(proxy)
library(mltools) 

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
