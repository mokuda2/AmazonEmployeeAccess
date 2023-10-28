library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)

amazon_train <- vroom("train.csv")
amazon_train$ACTION <- factor(amazon_train$ACTION)

amazon_test <- vroom("test.csv")

## knn
target_encoding_amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  #Everything numeric for SMOTE so encode it here
  step_smote(all_outcomes(), neighbors=5) %>%
  step_normalize()
prep <- prep(target_encoding_amazon_recipe)
baked_train <- bake(prep, new_data = amazon_train)

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(target_encoding_amazon_recipe) %>%
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
target_encoding_amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  #Everything numeric for SMOTE so encode it here
  step_smote(all_outcomes(), neighbors=5) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.9)
prep <- prep(target_encoding_amazon_recipe)
baked_train <- bake(prep, new_data = amazon_train)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(target_encoding_amazon_recipe) %>%
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
target_encoding_amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  #Everything numeric for SMOTE so encode it here
  step_smote(all_outcomes(), neighbors=5) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.9)
prep <- prep(target_encoding_amazon_recipe)
baked_train <- bake(prep, new_data = amazon_train)

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(target_encoding_amazon_recipe) %>%
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
