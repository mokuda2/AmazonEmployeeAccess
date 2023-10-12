library(tidyverse)
library(vroom)
library(tidymodels)

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
