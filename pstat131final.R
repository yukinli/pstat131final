library(corrplot)  # for the correlation plot
library(discrim)  # for linear discriminant analysis
library(corrr)   # for calculating correlation
library(knitr)   # to help with the knitting process
library(MASS)    # to assist with the markdown processes
library(tidyverse)   # using tidyverse and tidymodels for this project mostly
library(tidymodels)
library(ggplot2)   # for most of our visualizations
library(ggrepel)
library(ggimage)
library(rpart.plot)  # for visualizing trees
library(vip)         # for variable importance 
library(vembedr)     # for embedding links
library(janitor)     # for cleaning out our data
library(randomForest)   # for building our randomForest
library(stringr)    # for matching strings
library("dplyr")     # for basic r functions
library("yardstick") # for measuring certain metrics
library(RSQLite)
library(sqldf)
library(DBI)
library(ranger)
library(kknn)
tidymodels_prefer(1)
NBA<-read.csv("/Users/yukinli/Documents/finaldata_nba.csv") 
head(NBA)
dim(NBA) # getting the dimensions of our data
set.seed(1) # set seed
NBA<- NBA %>% # clean name
  clean_names()
NBA <- NBA %>%
  select(player,pos,age,mp,x3p_2,e_fg,ft,fta,ft_2,pts) %>%
  # ft_2 is free throw percentage, x3p is 3 points made,x3pa is 3 points attempts, x3p_2 is 3 points %.
  filter(fta != 0) 
# We do not want the players who have not attempted any free throw so we filter out fta = 0

NBA_split <- NBA %>% 
  initial_split(prop = 0.8, strata = "ft_2")
NBA_train <- training(NBA_split)
NBA_test <- testing(NBA_split)
dim(NBA_train) 
dim(NBA_test)
ggplot(NBA_train, aes(pos)) +
  geom_bar() +
  labs(
    title = "Number of Players of each Position",
    x = "Players' Position",
    y = "Count"
  ) +
  # We want to be able to read labels better
  coord_flip()
ggplot(NBA_train, aes(ft_2)) +
  geom_histogram(bins = 70, color = "white") +
  labs(
    title = "Histogram of Free Throws %"
  )

ggplot(NBA_train, aes(ft_2)) +
  geom_histogram(bins = 30, color = "white") +
  facet_wrap(~pos, scales = "free_y") +
  labs(
    title = "Histogram of Free Throws % by Players' Position"
  )
ggplot(NBA_train, aes(reorder(pos, ft_2), ft_2)) +
  geom_boxplot(varwidth = TRUE) + 
  coord_flip() +
  labs(
    title = " Free Throw % by Players' Position",
    x = "Players' Position"
  )

NBA_train %>% 
  ggplot(aes(fta, ft_2)) +
  geom_point(alpha = 0.1) +
  stat_summary(fun.y=mean, colour="red", geom="line", size = 3)+
  facet_wrap(~pos, scales = "free") +
  labs(
    title = "Free Throw Attempts vs. Free Throw % by Players' Postion"
  )
NBA_train %>% 
  ggplot(aes(ft, ft_2)) +
  geom_point(alpha = 0.1) +
  stat_summary(fun.y=mean, colour="red", geom="line", size = 3)+
  facet_wrap(~pos, scales = "free") +
  labs(
    title = "Free Throw Made vs. Free Throw % by Players' Postion"
  )
NBA_train %>% 
  group_by(pos, mp) %>% 
  summarize(
    mean_ft_2 = mean(ft_2)
  ) %>% 
  ggplot(aes(mp, mean_ft_2)) +
  geom_line() +
  geom_point() +
  facet_wrap(~pos) +
  labs(
    title = "Average Free Throw of Minutes Played per game by Players' Position",
    y = "Mean Free Throw %"
  )
NBA_train %>% 
  ggplot(aes(ft_2, x3p_2)) +
  geom_point(alpha = 0.1) +
  geom_smooth(se = FALSE, color = "red", size = 3) +
  facet_wrap(~pos, scales = "free_y") +
  labs(
    title = "Freee Throw % vs 3 Points % by Posotion"
  )
set.seed(1) # set seed
NBA<- NBA %>% # clean name
  clean_names()
NBA <- NBA %>%
  select(pos,mp,x3p_2,ft,fta,ft_2) %>%
  filter(fta != 0) 
NBA_split <- NBA %>% 
  initial_split(prop = 0.8, strata = "ft_2")
NBA_train <- training(NBA_split)
NBA_test <- testing(NBA_split)
NBA_folds <- vfold_cv(NBA_train, v = 5, strata = ft_2)
NBA_recipe <- recipe (
  ft_2 ~  pos + mp + x3p_2 + ft + fta , data = NBA_train) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_center(all_predictors()) %>% 
  step_scale(all_predictors())
rf_model <- 
  rand_forest(
    min_n = tune(),
    mtry = tune()) %>%
  set_mode ("regression") %>% 
  set_engine("ranger")
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(NBA_recipe)
rf_params <- hardhat::extract_parameter_set_dials(rf_model) %>% 
  update(mtry = mtry(range= c(2, 5)))
rf_grid <- grid_regular(rf_params, levels = 2)
rf_tune <- rf_workflow %>% 
  tune_grid(
    resamples = NBA_folds ,
    grid = rf_grid)
bt_model <- boost_tree(
  min_n = tune(),
  mtry = tune(),
  learn_rate = tune()) %>% 
  set_mode ( "regression")%>%
  set_engine("xgboost")
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(NBA_recipe)
bt_params <- hardhat::extract_parameter_set_dials(bt_model) %>% 
  update(mtry = mtry(range= c(2, 5)),
         learn_rate = learn_rate(range = c(-5, 0.2))
  )
bt_grid <- grid_regular(bt_params, levels = 2)
bt_tune <- bt_workflow %>% 
  tune_grid(
    resamples = NBA_folds, 
    grid = bt_grid
  )
nn_model <- 
  nearest_neighbor(
    neighbors = tune())%>%
  set_mode ("regression") %>% 
  set_engine("kknn")
nn_workflow <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(NBA_recipe)
nn_params <-hardhat::extract_parameter_set_dials(nn_model)
nn_grid <- grid_regular(nn_params, levels = 2)
nn_tune <- nn_workflow %>% 
  tune_grid(
    resamples = NBA_folds, 
    grid = nn_grid)
autoplot(rf_tune, metric = "rmse")
show_best(rf_tune, metric = "rmse") %>% select(-.estimator, -.config)
autoplot(bt_tune, metric = "rmse")
show_best(bt_tune, metric = "rmse") %>% select(-.estimator, -.config)
autoplot(nn_tune, metric = "rmse")
show_best(nn_tune, metric = "rmse") %>% select(-.estimator, -.config)
bt_workflow_tuned <- bt_workflow %>% 
  finalize_workflow(select_best(bt_tune, metric = "rmse"))
bt_results <- fit(bt_workflow_tuned, NBA_train)
NBA_metric <- metric_set(rmse)

model_test_predictions <- predict(bt_results, new_data = NBA_test) %>% 
  bind_cols(NBA_test %>% select(ft_2)) 

model_test_predictions_type <- predict(bt_results, new_data = NBA_test) %>% 
  bind_cols(NBA_test %>% select(ft_2, pos)) 

model_test_predictions %>% 
  NBA_metric(truth = ft_2, estimate = .pred)
ft_ex1<- data.frame(
  pos = 'C',
  mp = 40,
  x3p_2 = 0.4,
  ft = 2,
  fta =20
)

predict(bt_results, ft_ex1)
ft_ex2<- data.frame(
  pos = 'PG',
  mp = 30,
  x3p_2 = 0.8,
  ft = 8,
  fta =20
)

predict(bt_results, ft_ex2)
ft_ex3<- data.frame(
  pos = 'C',
  mp = 30,
  x3p_2 = 0.9,
  ft = 1,
  fta =20
)

predict(bt_results, ft_ex3)










