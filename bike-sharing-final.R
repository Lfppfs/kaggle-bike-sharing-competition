# loading packages
pkgs_to_load <- c("ggplot2", "car", "magrittr", "dplyr") # car for vif function
pkgs_loaded <- sapply(pkgs_to_load, library, character.only = TRUE)

train_data <- read.csv("/home/lfppfs/Desktop/Programming/kaggle/bike-sharing-competition/train.csv")
str(train_data)

# changing variables to factor
new_factor_vars <- c("season", "holiday", "workingday")
train_data %<>% mutate_at(new_factor_vars, ~(factor(.)))
train_data$datetime <- as.POSIXct(train_data$datetime)
str(train_data)

# potentially important visualizations
ggplot() + geom_point(data = train_data, aes(y = count, x = datetime, color = temp)) +
facet_grid(workingday ~ .) +
labs(title = "Rentals per datetime by work day") +
scale_colour_gradient2(midpoint = mean(train_data$temp), low = "blue", high = "red") +
scale_x_datetime(
    limits = c(min(train_data$datetime), max(train_data$datetime)),
    date_breaks = "2 month",
    date_labels = "%y/%b"
) +
theme(axis.text.x = element_text(angle = 45))

ggplot() + geom_boxplot(data = train_data, aes(y = count, x = as.factor(season)))

# this creates a variable with hours of the day
train_data$hour <- unclass(as.POSIXlt(train_data$datetime)$hour)

ggplot() + geom_point(data = train_data, aes(y = count, x = hour, color = temp),
    position=position_jitter(w=1, h=0)) +
facet_grid(workingday ~ .) +
labs(title = "Rentals per hour by work day") +
scale_colour_gradient2(midpoint = mean(train_data$temp), low = "blue", high = "red") +
scale_x_continuous(breaks = seq(0, 24, 2))

ggplot() + geom_point(data = train_data, aes(y = count, x = hour, color = temp),
    position=position_jitter(w=1, h=0)) +
facet_grid(holiday ~ .) +
labs(title = "Rentals per hour by holiday") +
scale_colour_gradient2(midpoint = mean(train_data$temp), low = "blue", high = "red") +
scale_x_continuous(breaks = seq(0, 24, 2))

# scatterplot matrix
pairs(select(train_data, !c("count", "atemp", "datetime")), lower.panel = NULL)

# not going to use casual and registered, since count
# is simply casual + registered
# temp and atemp are probably collinear, using vif to check
train_data_filtered <- train_data %>% select(!c(casual, registered))
bike_lm_v1 <- lm(count ~ ., train_data)
summary(bike_lm_v1)
vif(bike_lm_v1)

# high vif, discarding atemp
bike_lm_v1 <- update(bike_lm_v1, ~ . -atemp)
summary(bike_lm_v1)

# dropping datetime, dont think "day 1, hour 1" should be different
# from "day 19, hour 4", season and hour should deal with this better
bike_lm_v1 <- update(bike_lm_v1, ~ . - datetime)
summary(bike_lm_v1)

# checking to see effect of dropping these
train_data_filtered <- train_data %>% select(!c(atemp, datetime))
bike_lm_v2 <- lm(count ~ ., train_data_filtered)
summary(bike_lm_v2)


# weirdly, workingday doesnt explain rentals
# looking at the plot "Rentals per hour by work day", there is
# a difference between work and non-work day, but this
# difference seems to depend on the hour
# trying an interaction term
bike_lm_v2 <- update(bike_lm_v2, ~ . + workingday:hour)
summary(bike_lm_v2)
# now it makes more sense

# trying a polynomial for hour
bike_lm_hour_poly <- update(bike_lm_v2, ~ . + poly(hour,2))
summary(bike_lm_hour_poly)
# really low coefficient, abandoning idea
# besides its hard to understand how this would work along with
# the interaction workingday:hour

# looking at the plot "Rentals per hour by work day", we see
# that each year seems to be divided in two periods
# from ~may to ~sep, there are more bike rentals
# trying to incorporate this and see if predictions are better
train_data$mon <- unclass(as.POSIXlt(train_data$datetime)$mon)
train_data <- train_data %>% mutate(season = as.factor(if_else(mon %in% c(3:8) , "summer", "winter"))) # considering march through august as summer
ggplot() + geom_boxplot(data = train_data, aes(y = count, x = season))

train_data_filtered <- train_data %>% select(!c(atemp, casual, registered, datetime, mon))
bike_lm_v3 <- lm(count ~ . + workingday:hour, train_data_filtered)
summary(bike_lm_v3)
vif(bike_lm_v3) # interaction workingday:hour seems to have some collinearity

# changing the coding of season didnt improve the model much
# backwards selection
bike_lm_v3 <- update(bike_lm_v3, ~ . - windspeed)
summary(bike_lm_v3)

par(mfrow=c(2,2))
plot(bike_lm_v3)

# strong heteroscedasticity, trying transformation
bike_lm_v3 <- lm(log10(count) ~ . + workingday:hour - windspeed, train_data_filtered)
summary(bike_lm_v3)
plot(bike_lm_v3)
# plot looks better
# there is a point with high leverage
plot(hatvalues(bike_lm_v3))

# removing the point and fitting the model again
train_data_filtered <- train_data_filtered[-which.max(hatvalues(bike_lm_v3)),]
bike_lm_v3 <- lm(log10(count) ~ . + workingday:hour - windspeed, train_data_filtered)
summary(bike_lm_v3)
plot(bike_lm_v3)

# predicting test data
test_data <- read.csv("/home/lfppfs/Desktop/Programming/kaggle/bike-sharing-competition/test.csv")
str(test_data)
# applying same transformations done to train data to test data
test_data %<>% mutate_at(new_factor_vars, ~(factor(.)))
test_data$datetime <- as.POSIXct(test_data$datetime)
test_data$mon <- unclass(as.POSIXlt(test_data$datetime)$mon)
test_data <- test_data %>% mutate(season = as.factor(if_else(mon %in% c(3:8) , "summer", "winter"))) # considering march through august as summer
test_data$hour <- unclass(as.POSIXlt(test_data$datetime)$hour)
test_data_filtered <- test_data %>% select(!c(atemp, datetime, mon))

predicted_values <- predict(bike_lm_v3, test_data_filtered)

# calculating root mean squared logarithmic error (rmsle)
sum_differences <- 0
for(i in seq_along(predicted_values)){
    value <- (log(bike_lm_v3$fitted.values[i] + 1) - log(predicted_values[i] + 1))**2
    sum_differences <- sum_differences + value
}

(rmsle <- sqrt(sum_differences / length(bike_lm_v3$fitted.values)))

submission_sample <- read.csv("/home/lfppfs/Desktop/Programming/kaggle/bike-sharing-competition/sampleSubmission.csv")
submission_sample$count <- predicted_values
str(submission_sample)

write.csv(submission_sample, "/home/lfppfs/Desktop/Programming/kaggle/bike-sharing-competition/submission.csv",
    row.names = FALSE)
