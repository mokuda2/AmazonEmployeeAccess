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

ggplot(data = amazon_train, mapping = aes(x = ACTION)) +
  geom_bar()
  
ggplot(data = amazon_train, mapping = aes(x = ROLE_ROLLUP_1)) +
  geom_bar()

sorted_counts <- sort(table(amazon_train$ROLE_ROLLUP_1), decreasing = T)
sorted_counts_20 <- head(sorted_counts, 20)
df_counts_20 <- data.frame(Value = names(sorted_counts_20), Count = sorted_counts_20)
ggplot(df_counts_20, aes(x = Count.Var1, y = Count.Freq)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  scale_x_discrete(limits = rev(factor(df_counts_20$Value)))

amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_other(c(RESOURCE, MGR_ID, ROLE_ROLLUP_1, ROLE_ROLLUP_2, ROLE_DEPTNAME, ROLE_TITLE, ROLE_FAMILY_DESC, ROLE_FAMILY, ROLE_CODE), threshold=.01) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(amazon_recipe)
baked_train <- bake(prep, new_data = amazon_train)

## logistic regression
log_reg_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")

amazon_workflow <- workflow() %>%
add_recipe(amazon_recipe) %>%
add_model(log_reg_mod) %>%
fit(data = amazon_train) # Fit the workflow

amazon_predictions <- predict(amazon_workflow,
                                new_data=amazon_test,
                              type="prob") # "class" or "prob" (see doc)

amazon_predictions$Action <- if_else(amazon_predictions$.pred_1 >= .95, 1, 0)
amazon_predictions$Id <- amazon_test$id
amazon_final <- amazon_predictions %>%
  select(c(Id, Action))

write.csv(amazon_final, "./STAT\ 348/AmazonEmployeeAccess/logreg.csv", row.names = F)

## penalized logistic regression
target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  # step_other(all_nominal_predictors(), threshold=.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))
prep <- prep(target_encoding_amazon_recipe)
baked_train <- bake(prep, new_data = amazon_train)

pen_log_reg_model <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
  add_recipe(target_encoding_amazon_recipe) %>%
  add_model(pen_log_reg_model)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(amazon_train, v = 5, repeats=1)

# Run the CV
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize the Workflow & fit it
final_wf <-amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)

# Predict
amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type="prob")

amazon_predictions$Action <- amazon_predictions$.pred_1
amazon_predictions$Id <- amazon_test$id
amazon_final <- amazon_predictions %>%
  select(c(Id, Action))

write.csv(amazon_final, "./STAT\ 348/AmazonEmployeeAccess/penalized_log_reg.csv", row.names = F)

## random forest classification
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

## naive bayes
target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))
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

write.csv(amazon_final, "./STAT\ 348/AmazonEmployeeAccess/naivebayes.csv", row.names = F)

## knn
target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
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

write.csv(amazon_final, "./STAT\ 348/AmazonEmployeeAccess/knn.csv", row.names = F)

## pca
# naive bayes
target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
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

write.csv(amazon_final, "./STAT\ 348/AmazonEmployeeAccess/pcanaivebayes.csv", row.names = F)

# knn
target_encoding_amazon_recipe <- recipe(ACTION~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
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

write.csv(amazon_final, "./STAT\ 348/AmazonEmployeeAccess/pcaknn.csv", row.names = F)
