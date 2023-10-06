library(tidyverse)
library(vroom)
library(tidymodels)
library(ggmosaic)

## 112
amazon_train <- vroom("./STAT\ 348/AmazonEmployeeAccess/train.csv")
amazon_train

amazon_test <- vroom("./STAT\ 348/AmazonEmployeeAccess/test.csv")
amazon_test

ggplot(data = amazon_train, mapping = aes(x = ROLE_TITLE)) +
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
  step_other(RESOURCE, threshold=.01) %>%
  step_other(MGR_ID, threshold=.01) %>%
  step_other(ROLE_ROLLUP_1, threshold=.01) %>%
  step_other(ROLE_ROLLUP_2, threshold=.01) %>%
  step_other(ROLE_DEPTNAME, threshold=.01) %>%
  step_other(ROLE_TITLE, threshold=.01) %>%
  step_other(ROLE_FAMILY_DESC, threshold=.01) %>%
  step_other(ROLE_FAMILY, threshold=.01) %>%
  step_other(ROLE_CODE, threshold=.01) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(amazon_recipe)
baked_train <- bake(prep, new_data = amazon_train)
