library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)

amazon_train <- vroom("./STAT\ 348/AmazonEmployeeAccess/train.csv")
amazon_train$ACTION <- factor(amazon_train$ACTION)
amazon_train

amazon_test <- vroom("./STAT\ 348/AmazonEmployeeAccess/test.csv")
amazon_test

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create a workflow with model & recipe
target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))
prep <- prep(target_encoding_amazon_recipe)
baked_train <- bake(prep, new_data = amazon_train)

amazon_workflow <- workflow() %>%
  add_recipe(target_encoding_amazon_recipe) %>%
  add_model(rf_model)

# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1,(ncol(amazon_train) - 1))),
                            min_n(),
                            levels = 10) ## L^2 total tuning possibilities

# Set up K-fold CV
folds <- vfold_cv(amazon_train, v = 10, repeats=1)

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

write.csv(amazon_final, "./STAT\ 348/AmazonEmployeeAccess/rfclassification.csv", row.names = F)