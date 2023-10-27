library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)

amazon_train <- vroom("train.csv")
amazon_train$ACTION <- factor(amazon_train$ACTION)

amazon_test <- vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  #Everything numeric for SMOTE so encode it here
  step_smote(all_outcomes(), neighbors=5)

# apply the recipe to your data
prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = amazon_train)

## logistic regression
# log_reg_mod <- logistic_reg() %>% #Type of model
#   set_engine("glm")
# 
# amazon_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(log_reg_mod) %>%
#   fit(data = amazon_train) # Fit the workflow
# 
# amazon_predictions <- predict(amazon_workflow,
#                               new_data=amazon_test,
#                               type="prob") # "class" or "prob" (see doc)
# 
# amazon_predictions$Action <- if_else(amazon_predictions$.pred_1 >= .95, 1, 0)
# amazon_predictions$Id <- amazon_test$id
# amazon_final <- amazon_predictions %>%
#   select(c(Id, Action))
# 
# write.csv(amazon_final, "logreg.csv", row.names = F)
# 
# ## penalized logistic regression
# # target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
# #   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
# #   # step_other(all_nominal_predictors(), threshold=.001) %>%
# #   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))
# # prep <- prep(target_encoding_amazon_recipe)
# # baked_train <- bake(prep, new_data = amazon_train)
# 
# pen_log_reg_model <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
#   set_engine("glmnet")
# 
# amazon_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(pen_log_reg_model)
# 
# # Grid of values to tune over
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5) ## L^2 total tuning possibilities
# 
# # Split data for CV
# folds <- vfold_cv(amazon_train, v = 5, repeats=1)
# 
# # Run the CV
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc)) #Or leave metrics NULL
# 
# # Find Best Tuning Parameters
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# # Finalize the Workflow & fit it
# final_wf <-amazon_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_train)
# 
# # Predict
# amazon_predictions <- final_wf %>%
#   predict(new_data = amazon_test, type="prob")
# 
# amazon_predictions$Action <- amazon_predictions$.pred_1
# amazon_predictions$Id <- amazon_test$id
# amazon_final <- amazon_predictions %>%
#   select(c(Id, Action))
# 
# write.csv(amazon_final, "penalized_log_reg.csv", row.names = F)

## random forest classification
rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create a workflow with model & recipe
# target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))
# prep <- prep(target_encoding_amazon_recipe)
# baked_train <- bake(prep, new_data = amazon_train)

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_model)

# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1,(ncol(amazon_train) - 1))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

# Set up K-fold CV
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type="prob")

amazon_predictions$Action <- amazon_predictions$.pred_1
amazon_predictions$Id <- amazon_test$id
amazon_final <- amazon_predictions %>%
  select(c(Id, Action))

write.csv(amazon_final, "rfclassification.csv", row.names = F)

## naive bayes
# target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))
# prep <- prep(target_encoding_amazon_recipe)
# baked_train <- bake(prep, new_data = amazon_train)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# Tune smoothness and Laplace here
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

# Set up K-fold CV
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Predict
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type="prob")

amazon_predictions$Action <- amazon_predictions$.pred_1
amazon_predictions$Id <- amazon_test$id
amazon_final <- amazon_predictions %>%
  select(c(Id, Action))

write.csv(amazon_final, "naivebayes.csv", row.names = F)

## knn
# target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
#   step_normalize()
# prep <- prep(target_encoding_amazon_recipe)
# baked_train <- bake(prep, new_data = amazon_train)

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

# Fit or Tune Model HERE
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(amazon_train, v = 5, repeats=1)

CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Predict
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type="prob")

amazon_predictions$Action <- amazon_predictions$.pred_1
amazon_predictions$Id <- amazon_test$id
amazon_final <- amazon_predictions %>%
  select(c(Id, Action))

write.csv(amazon_final, "knn.csv", row.names = F)

## pca
# naive bayes
# target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
#   step_normalize(all_predictors()) %>%
#   step_pca(all_predictors(), threshold=.9)
# prep <- prep(target_encoding_amazon_recipe)
# baked_train <- bake(prep, new_data = amazon_train)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# Tune smoothness and Laplace here
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

# Set up K-fold CV
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Predict
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type="prob")

amazon_predictions$Action <- amazon_predictions$.pred_1
amazon_predictions$Id <- amazon_test$id
amazon_final <- amazon_predictions %>%
  select(c(Id, Action))

write.csv(amazon_final, "pcanaivebayes.csv", row.names = F)

# knn
# target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
#   step_normalize(all_predictors()) %>%
#   step_pca(all_predictors(), threshold=.9)
# prep <- prep(target_encoding_amazon_recipe)
# baked_train <- bake(prep, new_data = amazon_train)

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

# Fit or Tune Model HERE
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(amazon_train, v = 5, repeats=1)

CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Predict
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type="prob")

amazon_predictions$Action <- amazon_predictions$.pred_1
amazon_predictions$Id <- amazon_test$id
amazon_final <- amazon_predictions %>%
  select(c(Id, Action))

write.csv(amazon_final, "pcaknn.csv", row.names = F)

## svm
svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

# target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
#   step_normalize()
# prep <- prep(target_encoding_amazon_recipe)
# baked_train <- bake(prep, new_data = amazon_train)

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

# Fit or Tune Model HERE
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 2) ## L^2 total tuning possibilities

folds <- vfold_cv(amazon_train, v = 2, repeats=1)

CV_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Predict
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

amazon_predictions <- predict(final_wf, new_data = amazon_test, type="prob")

amazon_predictions$Action <- amazon_predictions$.pred_1
amazon_predictions$Id <- amazon_test$id
amazon_final <- amazon_predictions %>%
  select(c(Id, Action))

write.csv(amazon_final, "svmradial.csv", row.names = F)






## xgboost
# amazon_train <- vroom("./STAT\ 348/AmazonEmployeeAccess/train.csv")
# amazon_train$ACTION <- factor(amazon_train$ACTION)
# 
# amazon_test <- vroom("./STAT\ 348/AmazonEmployeeAccess/test.csv")
# 
# # Define XGBoost model
# xgb_model <- boost_tree(
#   trees = tune(),
#   tree_depth = tune(),
#   mtry = tune(),
#   learn_rate = tune(),
#   loss_reduction = tune(),
#   sample_size = tune(),
#   stop_iter = 50  # Define a stopping criterion
# ) %>%
#   set_engine("xgboost", objective = "binary:logistic") %>%
#   set_mode("classification")
# 
# # Create a workflow with model & recipe
# target_encoding_amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   step_normalize()
# prep <- prep(target_encoding_amazon_recipe)
# baked_train <- bake(prep, new_data = amazon_train)
# 
# amazon_workflow <- workflow() %>%
#   add_recipe(target_encoding_amazon_recipe) %>%
#   add_model(xgb_model)
# 
# # Set up grid of tuning values
# tuning_grid <- grid_regular(
#   trees(range = c(900, 1000)),  # Adjust the range as per your preference
#   tree_depth(range = c(3, 10)),
#   mtry(range = c(1, (ncol(amazon_train) - 1))),
#   learn_rate(range = c(0.01, 0.1)),
#   loss_reduction(range = c(0, 1)),
#   sample_size(range = c(0, 1))
# )
# 
# # Set up K-fold CV
# folds <- vfold_cv(amazon_train, v = 10, repeats = 1)
# 
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples = folds, grid = tuning_grid, metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and predict
# final_wf <- amazon_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = amazon_train)
# 
# amazon_predictions <- final_wf %>%
#   predict(new_data = amazon_test, type = "prob")
# 
# amazon_predictions$Action <- amazon_predictions$.pred_1
# amazon_predictions$Id <- amazon_test$id
# amazon_final <- amazon_predictions %>%
#   select(c(Id, Action))
# 
# write.csv(amazon_final, "./STAT\ 348/AmazonEmployeeAccess/xgboost_classification.csv", row.names = FALSE)
# 
# ## imbalanced data
# my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   #Everything numeric for SMOTE so encode it here
#   step_smote(all_outcomes(), neighbors=5)
# 
# # apply the recipe to your data
# prepped_recipe <- prep(my_recipe)
# baked <- bake(prepped_recipe, new_data = amazon_train)
