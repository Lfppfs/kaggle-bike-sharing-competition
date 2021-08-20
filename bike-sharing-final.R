# loading packages
pkgs_to_load <- c("ggplot2", "car", "magrittr", "dplyr", "caTools", "Metrics", "AER",
    "FNN", "rpart", "mgcv", "randomForest")
pkgs_loaded <- sapply(pkgs_to_load, library, character.only = TRUE)

data <- read.csv("/home/lfppfs/Desktop/Programming/kaggle/bike-sharing-competition/train.csv")
str(data)

set.seed(1234)
sample <- caTools::sample.split(data, 0.8)
train_data <- subset(data, sample == TRUE)
test_data <- subset(data, sample == FALSE)

# changing variables to factor
new_factor_vars <- c("season", "holiday", "workingday")
train_data %<>% mutate_at(new_factor_vars, ~(factor(.)))
train_data$datetime <- as.POSIXct(train_data$datetime)
str(train_data)

# potentially important visualizations
ggplot() + geom_point(data = train_data, aes(
    y = count, x = datetime, color = temp)) +
facet_grid(workingday ~ .) +
labs(title = "Rentals per datetime by work day") +
scale_colour_gradient2(midpoint = mean(train_data$temp),
    low = "blue", high = "red") +
scale_x_datetime(
    limits = c(min(train_data$datetime), max(train_data$datetime)),
    date_breaks = "2 month",
    date_labels = "%y/%b"
) +
theme(axis.text.x = element_text(angle = 45))

ggplot() + geom_boxplot(data = train_data, aes(
    y = count, x = as.factor(season)))

# this creates a variable with hours of the day
train_data$hour <- unclass(as.POSIXlt(train_data$datetime)$hour)

ggplot() + geom_point(data = train_data, aes(
    y = count, x = hour, color = temp), position=position_jitter(w=1, h=0)) +
facet_grid(workingday ~ .) +
labs(title = "Rentals per hour by work day") +
scale_colour_gradient2(midpoint = mean(train_data$temp),
    low = "blue", high = "red") +
scale_x_continuous(breaks = seq(0, 24, 2))

ggplot() + geom_point(data = train_data, aes(
    y = count, x = hour, color = temp), position=position_jitter(w=1, h=0)) +
facet_grid(holiday ~ .) +
labs(title = "Rentals per hour by holiday") +
scale_colour_gradient2(midpoint = mean(train_data$temp),
    low = "blue", high = "red") +
scale_x_continuous(breaks = seq(0, 24, 2))

# scatterplot matrix
pairs(select(train_data, !c("count", "atemp", "datetime")), lower.panel = NULL)

# not going to use casual and registered, since count
# is simply casual + registered
train_data_filtered <- train_data %>% select(!c(casual, registered))

# linear models
bike_lm_v1 <- lm(count ~ ., train_data_filtered)
summary(bike_lm_v1)

# temp and atemp are probably collinear, using vif to check
car::vif(bike_lm_v1)
# high vif, discarding atemp
bike_lm_v1 <- update(bike_lm_v1, ~ . -atemp)
summary(bike_lm_v1)

# dropping datetime, dont think "day 1, hour 1" should be different
# from "day 19, hour 4", season and hour should deal with this better
bike_lm_v1 <- update(bike_lm_v1, ~ . - datetime)
summary(bike_lm_v1)
# this model has a serious issue, some of its fitted values are negative:
any(bike_lm_v1$fitted.values < 0)
# this is solved below when a log10 transformation is applied to the
# response variable

# weirdly, workingday doesnt explain rentals
# looking at the plot "Rentals per hour by work day", there is
# a difference between work and non-work day, but this
# difference seems to depend on the hour
# trying an interaction term
bike_lm_v2 <- update(bike_lm_v1, ~ . + workingday:hour)
summary(bike_lm_v2)
# now it makes more sense

# trying a polynomial for hour
bike_lm_hour_poly <- update(bike_lm_v2, ~ . + poly(hour,2))
summary(bike_lm_hour_poly)
# really low coefficient, abandoning idea
# besides its hard to understand how this would work along with
# the interaction workingday:hour

# looking at the plot "Rentals per datetime by work day", we see
# that each year seems to be divided in two periods
# from ~may to ~sep, there are more bike rentals
# trying to incorporate this and see if predictions are better
train_data_filtered$mon <- unclass(as.POSIXlt(train_data_filtered$datetime)$mon)
train_data_filtered_season_mutated <- train_data_filtered %>% mutate(season = as.factor(
     # considering march through august as summer
    if_else(mon %in% c(3:8) , "summer", "winter")))
ggplot() + geom_boxplot(data = train_data_filtered, aes(y = count, x = season))
train_data_filtered_season_mutated <- train_data_filtered_season_mutated %>%
    select(!c(atemp))
bike_lm_v3 <- lm(count ~ . + workingday:hour - datetime, train_data_filtered_season_mutated)
summary(bike_lm_v3)
# changing the coding of season didnt improve the model much
# back to original model

str(train_data_filtered)
train_data_filtered <- train_data_filtered %>% select(!c(atemp))
bike_lm_v4 <- lm(count ~ . + workingday:hour - datetime, train_data_filtered)
summary(bike_lm_v4)

# backwards selection
bike_lm_v4 <- update(bike_lm_v4, ~ . - windspeed)
summary(bike_lm_v4)

bike_lm_v4 <- update(bike_lm_v4, ~ . - holiday)
summary(bike_lm_v4)

# visualizing model fit to assumptions
par(mfrow=c(2,2))
plot(bike_lm_v4)

# strong heteroscedasticity, trying transformation
bike_lm_v4_log10 <- update(bike_lm_v4, log10(count) ~ .)
summary(bike_lm_v4_log10)
plot(bike_lm_v4_log10)
# plot looks better, although now weather does not
# influence count
# apparently no points with high leverage
plot(hatvalues(bike_lm_v4_log10))

# applying same transformations done to train data to test data
test_data %<>% mutate_at(new_factor_vars, ~(factor(.)))
test_data$datetime <- as.POSIXct(test_data$datetime)
test_data$mon <- unclass(as.POSIXlt(test_data$datetime)$mon)
test_data$hour <- unclass(as.POSIXlt(test_data$datetime)$hour)
test_data_filtered <- test_data %>% select(!c(atemp))

predicted_values <- predict(bike_lm_v4_log10, test_data_filtered)
rmsle(test_data_filtered$count, predicted_values)

# GLM
# a poisson is adequate for modeling this regression, since count is a positive integer and there will not be any coefficients < 0
bike_poisson_glm <- glm(count ~ . - datetime, family = 'poisson',
    data = train_data_filtered)
summary(bike_poisson_glm)
# checking for overdispersion
AER::dispersiontest(bike_poisson_glm)
# model has overdispersion
# this can be dealt with but I'm abandoning the idea
# the glm function does not return an r^2, and I don't think there is a consensus on how to measure goodness-of-fit for a glm. Dealing with the overdispersion of the model seems like too much trouble, not worth it

# GAMs
gam_v1 <- gam(count ~ season + workingday + holiday + weather +
    s(temp) + humidity + hour, data = train_data_filtered)
summary(gam_v1)
gam_v2 <- gam(count ~ season + workingday + holiday + weather +
    s(temp) + s(humidity) + hour, data = train_data_filtered)
summary(gam_v2)
# the model below is the best one can do with, since using smooth
# functions with the other variables returns an error
gam_v3 <- gam(count ~ season + workingday + holiday + weather + s(temp) +
    s(humidity) + s(hour), data = train_data_filtered)
summary(gam_v3)
any(gam_v3$fitted.values < 0)
# has fitted values < 0

# changing to poisson to account for fitted values < 0
gam_v4 <- gam(count ~ season + workingday + holiday + weather + s(temp) +
    s(humidity) + s(hour), family = 'poisson', data = train_data_filtered)
summary(gam_v4)
any(gam_v4$fitted.values < 0)

# plotting the model returns plots of fitted
# values (y axis) versus smoothed predictors
plot(gam_v4)
# we can see that bike rentals increase in a curve
# with temp and humidity (makes sense)
# and bike rentals change in a highly nonlinear fashion with hours (also
# resembles the plot rentals per hour by work day)

predicted_values <- predict(gam_v4, test_data_filtered)
rmsle(test_data_filtered$count, predicted_values)
# has lower rmsle than linear model
# using data with seasons = c(summer, winter) lowers the fit of the model,
# giving a rmsle = 3.17 (not shown here)

# using interaction hour:workingday
gam_v5 <- gam(count ~ season + workingday + holiday + weather + s(temp) +
    s(humidity) + s(hour) + hour:workingday, family = 'poisson',
    data = train_data_filtered)
summary(gam_v5)
predicted_values <- predict(gam_v5, test_data_filtered)
rmsle(test_data_filtered$count, predicted_values)
# doesnt calculate intercept and also lowers fit

# decision trees
tree_v1 <- rpart(count ~ season, method = 'anova', train_data_filtered)
tree_v1
summary(tree_v1)
attributes(tree_v1)
tree_v1$cptable

tree_v2 <- rpart(count ~ season + temp, method = 'anova', train_data_filtered)
tree_v2
summary(tree_v2)
str(train_data_filtered)

tree_v3 <- rpart(count ~ ., method = 'anova', train_data_filtered)
tree_v3
summary(tree_v3)
attributes(tree_v3)

tree_v3$cptable
sort(tree_v3$variable.importance, decreasing = TRUE)
plot(tree_v3)
text(tree_v3)

predicted_values <- predict(tree_v3, test_data_filtered)
rmsle(test_data_filtered$count, predicted_values)
# much lower rmsle than in linear regression

# excluding datetime
tree_v4 <- rpart(count ~ . - datetime, method = 'anova', train_data_filtered)
tree_v4
summary(tree_v4)
attributes(tree_v4)

tree_v4$cptable
sort(tree_v4$variable.importance, decreasing = TRUE)
plot(tree_v4)
text(tree_v4)

predicted_values <- predict(tree_v4, test_data_filtered)
rmsle(test_data_filtered$count, predicted_values)
# almost no difference in rmsle
# datetime is very important in this model so I don't think
# excluding it is really a good move

# random forests
bike_forest <- randomForest(count ~ ., train_data_filtered)
bike_forest
# 90% of variance explained
importance_df <- data.frame(importance = bike_forest$importance[1:10],
    variable = dimnames(bike_forest$importance)[[1]])
importance_df
arrange(importance_df, desc(importance))
varImpPlot(bike_forest)
# most important variables are hour and datetime

forest_predictions <- predict(bike_forest, test_data_filtered)
rmsle(test_data_filtered$count, forest_predictions)
# lower than decision tree

# knn
# knn.reg from package FNN only accepts integers as predictors
new_factor_vars <- c("season", "holiday", "workingday")
train_knn <- train_data_filtered
train_knn %<>% mutate_at(new_factor_vars, ~(as.integer(.)))

# running knn with k = [3,20] and gathering best model
# for training set only first
knn_results = vector("list", 17)
best_model = list(PRESS = Inf,
    model = NULL)
for (k in 3:20){
    knn_results[[k]] <- knn.reg(train_knn[, !colnames(train_knn) %in%
        c("count", "datetime")], y = train_knn$count, k = k)
    if (knn_results[[k]]$PRESS < best_model$PRESS) {
        best_model$PRESS = knn_results[[k]]$PRESS
        best_model$model = knn_results[[k]]
    }

}
# best_model$model
str(best_model)
# best_model$model$k # k of best model

# predicting test set accuracy using k from training set
new_factor_vars <- c("season", "holiday", "workingday")
test_knn <- test_data_filtered
test_knn %<>% mutate_at(new_factor_vars, ~(as.integer(.)))
test_knn <- test_knn %>% select(!c(casual, registered))
bike_knn <- knn.reg(
    train = train_knn[, !colnames(train_knn) %in%
    c("count", "datetime", "atemp")],
    test = test_knn[, !colnames(test_knn) %in%
    c("count", "datetime", "atemp")], y = train_knn$count,
    k = best_model$model$k)
best_model$PRESS = 1
bike_knn

# plotting actual ~ predicted
ggplot() + geom_point(aes(x = test_data_filtered$count, y = bike_knn$pred)) +
    labs(x = "Actual test values", y = "Predicted values")

rmsle(test_data_filtered$count, bike_knn$pred)
# higher than random forest

# CONCLUSION
# best prediction model came from random forest
