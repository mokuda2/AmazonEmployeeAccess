library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)

amazon_train <- vroom("train.csv")
amazon_train$ACTION <- factor(amazon_train$ACTION)
amazon_train

amazon_test <- vroom("test.csv")
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

write.csv(amazon_final, "logreg.csv", row.names = F)

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

write.csv(amazon_final, "penalized_log_reg.csv", row.names = F)
