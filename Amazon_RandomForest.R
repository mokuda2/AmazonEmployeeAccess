# library(tidyverse)
# library(vroom)
# library(tidymodels)
# library(embed)
# library(discrim)
# library(naivebayes)
# library(kknn)
# 
# amazon_train <- vroom("train.csv")
# amazon_train$ACTION <- factor(amazon_train$ACTION)
# amazon_train
# 
# amazon_test <- vroom("test.csv")
# amazon_test
# 
# rf_model <- rand_forest(mtry = tune(),
#                         min_n = tune(),
#                         trees=1000) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create a workflow with model & recipe
# target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))
# prep <- prep(target_encoding_amazon_recipe)
# baked_train <- bake(prep, new_data = amazon_train)
# 
# amazon_workflow <- workflow() %>%
#   add_recipe(target_encoding_amazon_recipe) %>%
#   add_model(rf_model)
# 
# # Set up grid of tuning values
# tuning_grid <- grid_regular(mtry(range=c(1,(ncol(amazon_train) - 1))),
#                             min_n(),
#                             levels = 10) ## L^2 total tuning possibilities
# 
# # Set up K-fold CV
# folds <- vfold_cv(amazon_train, v = 10, repeats=1)
# 
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc)) #Or leave metrics NULL
# 
# # Find best tuning parameters
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and predict
# final_wf <- amazon_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_train)
# 
# amazon_predictions <- final_wf %>%
#   predict(new_data = amazon_test, type="prob")
# 
# amazon_predictions$Action <- amazon_predictions$.pred_1
# amazon_predictions$Id <- amazon_test$id
# amazon_final <- amazon_predictions %>%
#   select(c(Id, Action))
# 
# write.csv(amazon_final, "rfclassification.csv", row.names = F)

library(tidyverse)
library(vroom)
library(tidymodels)
library(xgboost)  # Add XGBoost library

# Load data
amazon_train <- vroom("train.csv")
amazon_train$ACTION <- factor(amazon_train$ACTION)

amazon_test <- vroom("test.csv")

# Define XGBoost model
xgb_model <- boost_tree(
  trees = tune(), 
  tree_depth = tune(),
  mtry = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 50  # Define a stopping criterion
) %>%
  set_engine("xgboost", objective = "binary:logistic") %>%
  set_mode("classification")

# Create a workflow with model & recipe
target_encoding_amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
prep <- prep(target_encoding_amazon_recipe)
baked_train <- bake(prep, new_data = amazon_train)

amazon_workflow <- workflow() %>%
  add_recipe(target_encoding_amazon_recipe) %>%
  add_model(xgb_model)

# Set up grid of tuning values
tuning_grid <- grid_regular(
  trees(range = c(100, 1000)),  # Adjust the range as per your preference
  tree_depth(range = c(3, 10)),
  mtry(range = c(1, (ncol(amazon_train) - 1))),
  learn_rate(range = c(0.01, 0.1)),
  loss_reduction(range = c(0, 1)),
  sample_size(range = c(0.5, 1))
)

# Set up K-fold CV
folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

CV_results <- amazon_workflow %>%
  tune_grid(resamples = folds, grid = tuning_grid, metrics = metric_set(roc_auc))

# Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type = "prob")

amazon_predictions$Action <- amazon_predictions$.pred_1
amazon_predictions$Id <- amazon_test$id
amazon_final <- amazon_predictions %>%
  select(c(Id, Action))

write.csv(amazon_final, "xgboost_classification.csv", row.names = FALSE)
