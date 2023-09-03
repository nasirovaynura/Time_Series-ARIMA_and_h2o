library(tidyverse)
library(tidymodels)
library(inspectdf)
library(data.table)
library(modeltime)
library(skimr)
library(timetk)
library(highcharter)
library(h2o)
library(forecast)


df <- read_csv("daily-minimum-temperatures-in-me.csv"); view(df)

glimpse(df)
inspect_na(df)

names(df) <- names(df) %>% gsub(' ', '_', .)

df$Daily_minimum_temperatures <- parse_number(df$Daily_minimum_temperatures)

df$Date <- df$Date %>% as.Date("%m/%d/%Y")

df %>% 
  plot_time_series(
    Date, Daily_minimum_temperatures, 
    .color_var = lubridate::year(Date),
    .color_lab = "Year",
    .interactive = T,
    .plotly_slider = T,
    .smooth = F)


# Seasonality plot

df %>%
  plot_seasonal_diagnostics(
    Date, Daily_minimum_temperatures, .interactive = T)


# 1. Build h2o::automl(). ----

h2o.init()    

df <- df %>% tk_augment_timeseries_signature(Date) %>% select(Daily_minimum_temperatures, everything())

df %>% skim()

df <- df %>%
  select(-contains("hour"),
         -contains("day"),
         -contains("week"),
         -minute,-second,-am.pm) %>% 
  mutate_if(is.ordered, as.character) %>% 
  mutate_if(is.character,as_factor)


h2o_data <- df %>% as.h2o()

train <- df %>% filter(year < 1988) %>% as.h2o()
test <- df %>% filter(year >= 1988) %>% as.h2o()

target <- df[, 1] %>% names()
features <- df[, -1] %>% names()

model <- h2o.automl(
  x = features, y = target, 
  training_frame = train, 
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "RMSE",
  exclude_algos = c("GLM", "GBM", "DRF", "XGBoost"),
  seed = 123, nfolds = 10,
  max_runtime_secs = 360) 

model@leaderboard %>% as.data.frame() %>% view()
model <- model@leader

y_pred <- model %>% h2o.predict(test) %>% as.data.frame() 
y_pred$predict


model %>% 
  h2o.rmse(train = T,
           valid = T,
           xval = T)

error_tbl <- df %>% 
  filter(lubridate::year(Date) >= 1988) %>% 
  add_column(pred = y_pred %>% as_tibble() %>% pull(predict)) %>%
  rename(actual = Daily_minimum_temperatures) %>% 
  select(Date, actual, pred)

highchart() %>% 
  hc_xAxis(categories = error_tbl$Date) %>% 
  hc_add_series(data = error_tbl$actual, type = 'line', color = 'red', name = 'Actual') %>% 
  hc_add_series(data = error_tbl$pred, type = 'line', color = 'green', name = 'Predicted') %>% 
  hc_title(text = 'Predict')


# New data (next years) 

new_data <- seq(as.Date("1991/01/01"), as.Date("1991/12/01"), "months") %>%
  as_tibble() %>% 
  add_column(Daily_minimum_temperatures = 0) %>% 
  rename(Date = value) %>% 
  tk_augment_timeseries_signature() %>%
  select(-contains("hour"),
         -contains("day"),
         -contains("week"),
         -minute, -second, -am.pm) %>% 
  mutate_if(is.ordered, as.character) %>% 
  mutate_if(is.character, as_factor)


# Forecast 

new_h2o <- new_data %>% as.h2o()

new_predictions <- model %>% 
  h2o.predict(new_h2o) %>% 
  as_tibble() %>%
  add_column(Date = new_data$Date) %>% 
  select(Date, predict) %>% 
  rename(Daily_minimum_temperatures = predict)

df %>% 
  bind_rows(new_predictions) %>% 
  mutate(colors = c(rep('Actual', 3650), rep('Predicted', 12))) %>% 
  hchart("line", hcaes(Date, Daily_minimum_temperatures, group = colors)) %>% 
  hc_title(text = 'Forecast') %>% 
  hc_colors(colors = c('red', 'green'))
 

# Model evaluation ----

test_set <- test %>% as.data.frame()
residuals = test_set$Daily_minimum_temperatures - y_pred$predict

RMSE = sqrt(mean(residuals^2))

y_test_mean = mean(test_set$Daily_minimum_temperatures)

tss = sum((test_set$Daily_minimum_temperatures - y_test_mean)^2)
rss = sum(residuals^2)

R2 = 1 - (rss/tss); R2

n <- test_set %>% nrow() 
k <- features %>% length()
Adjusted_R2 = 1 - (1 - R2) * ((n-1) / (n-k-1))

tibble(RMSE = round(RMSE),
       R2, Adjusted_R2)


# 2. Build modeltime::arima_reg(). For this task set engine to “auto_arima” ----

splits <- initial_time_split(df, prop = 0.8)

model_fit_arima_no_boost <- arima_reg() %>%
  set_engine(engine = "auto_arima") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))


# Additional models

model_fit_arima_boosted <- arima_boost(
  min_n = 2,
  learn_rate = 0.015) %>% 
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))

model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))

model_fit_prophet <- prophet_reg(seasonality_daily = T) %>%
set_engine(engine = "prophet") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))


recipe_spec <- recipe(Daily_minimum_temperatures ~ Date, data = training(splits)) %>%
  step_date(Date, features = "month", ordinal = FALSE) %>%
  step_mutate(date_num = as.numeric(date)) %>%
  step_normalize(date_num) %>%
  step_rm(Date)

wflw_fit_mars <- workflow() %>%
  add_recipe(recipe_spec) %>%
  add_model(model_spec_mars) %>%
  fit(training(splits))

models_tbl <- modeltime_table(
  model_fit_arima_no_boost,
  model_fit_arima_boosted,
  model_fit_ets,
  model_fit_prophet)

models_tbl


# Calibration

calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl

calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = df) %>% 
  plot_modeltime_forecast(
    .legend_max_width = 25, 
    .interactive      = T)


calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = T)

# 3. Forecast temperatures for next year with model which has lower RMSE. ----

calibration_tbl <- model_fit_prophet %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl %>%
  modeltime_forecast(h = "1 year", 
                     new_data = testing(splits),
                     actual_data = df) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, 
    .interactive      = T)


