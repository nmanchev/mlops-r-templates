# Databricks notebook source
# MAGIC %md
# MAGIC # Train Model R Template Notebook
# MAGIC
# MAGIC ## Notebook setup
# MAGIC
# MAGIC We start by reading the values from all notebook parameters. We also load all R packages that are required for the notebook (missing packages will be installed on the fly)

# COMMAND ----------

# Fully qualified name of the UC-registered Delta table containing the training data.
dbutils.widgets.text("training_data_path", "nmanchev.mlops_template.iris", "Training table")

# MLflow registered model name to use for the trained mode.
dbutils.widgets.text("model_name", "dev-my_mlops_project-model", label="Model Name")

# MLflow experiment name.
dbutils.widgets.text("experiment_name", "/dev-my_mlops_project-experiment", label="MLflow experiment name")

# Baseline table
dbutils.widgets.dropdown("monitoring_mode", "enabled", list("enabled", "disabled"), label="Monitoring Mode")

# UC catalog/schema for model registration
dbutils.widgets.text("uc_catalog", "main", label="UC Catalog")
dbutils.widgets.text("uc_schema", "nmanchev", label="UC Schema")

# Baseline table
dbutils.widgets.text("baseline_table", "baseline_table", label="Baseline Table")

training_data_path <- dbutils.widgets.get("training_data_path")
model_name <- dbutils.widgets.get("model_name")
experiment_name <- dbutils.widgets.get("experiment_name")
monitoring_mode <- dbutils.widgets.get("monitoring_mode")


# COMMAND ----------

# Install all packages needed in this notebook

# Package names
packages <- c("dplyr", "xgboost", "sparklyr", "mlflow")

# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Packages loading
invisible(lapply(packages, library, character.only = TRUE))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, the notebook tries to load the demo data from the UC table. If this table doesn't exist we can create it by running the following code:
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC library(sparklyr)
# MAGIC sc <- spark_connect(method = "databricks")
# MAGIC tbl_iris <- sdf_copy_to(sc, iris, "iris", overwrite = TRUE)
# MAGIC
# MAGIC sparklyr::spark_write_table( 
# MAGIC   x = tbl_iris, 
# MAGIC   name = training_data_path, 
# MAGIC   mode = "overwrite"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading

# COMMAND ----------

# Load the training data from the UC table and display its contents
sc <- spark_connect(method = "databricks")

iris_df <- collect(dplyr::tbl(sc, training_data_path))
display(iris_df)

# COMMAND ----------

summary(iris_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation
# MAGIC
# MAGIC Next, we split the data into training and test sets.

# COMMAND ----------

# 75% of the sample size
smp_size <- floor(0.75 * nrow(iris_df))

# set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(iris_df)), size = smp_size)

# We need to convert the class label to a 0-indexed integer variable as required by XGBoost
species = iris$Species
label = as.integer(iris$Species)-1
iris$Species = NULL

train_data = as.matrix(iris[train_ind,])
train_label = label[train_ind]
test_data = as.matrix(iris[-train_ind,])
test_label = label[-train_ind]

# COMMAND ----------

cat("Number of samples in the training set:", nrow(train_data), "\n")
cat("Number of samples in the test set.   :", nrow(test_data))

# COMMAND ----------

xgb_train = xgb.DMatrix(data=train_data,label=train_label)
xgb_test = xgb.DMatrix(data=test_data,label=test_label)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Model training

# COMMAND ----------

mlflow_log_model <- function(model, artifact_path, signature = NULL, ...) {
  
  format_signature <- function(signature) {
    lapply(signature, function(x) {
      jsonlite::toJSON(x, auto_unbox = TRUE)
    })
  }
  
  temp_path <- fs::path_temp(artifact_path)
  
  model_spec <- mlflow_save_model(model, path = temp_path, model_spec = list(
    utc_time_created = mlflow:::mlflow_timestamp(),
    run_id = mlflow:::mlflow_get_active_run_id_or_start_run(),
    artifact_path = artifact_path, 
    flavors = list(),
    signature = format_signature(signature)
  ), ...)
  
  res <- mlflow_log_artifact(path = temp_path, artifact_path = artifact_path)
  
  tryCatch({
    mlflow:::mlflow_record_logged_model(model_spec)
  },
  error = function(e) {
    warning(
      paste("Logging model metadata to the tracking server has failed, possibly due to older",
            "server version. The model artifacts have been logged successfully.",
            "In addition to exporting model artifacts, MLflow clients 1.7.0 and above",
            "attempt to record model metadata to the  tracking store. If logging to a",
            "mlflow server via REST, consider  upgrading the server version to MLflow",
            "1.7.0 or above.", sep=" ")
    )
  })
  res
}

# overriding the function in the existing mlflow namespace 
assignInNamespace("mlflow_log_model", mlflow_log_model, ns = "mlflow")

# COMMAND ----------

extract_schema <- function(df) {
  schema <- lapply(names(df), function(col_name) {
    list(type = typeof(df[[col_name]])[1], name = col_name)
  })
  return(schema)
}


input_schema <- extract_schema(iris)
signature <- list(
    inputs = input_schema,
    outputs = list(list(type = "double")))


# COMMAND ----------

mlflow_set_experiment(experiment_name)

xgb_predict <- function(model, test_data) {

  ## Generate prediction on the test data
  xgb_pred = predict(model,test_data,reshape=T)
  xgb_pred = as.data.frame(xgb_pred)
  colnames(xgb_pred) = levels(species)

  xgb_pred$prediction = apply(xgb_pred,1,function(x) colnames(xgb_pred)[which.max(x)])
  xgb_pred$label = levels(species)[test_label+1]

  xgb_pred
}

train_model <- function(eta=0.001, max_depth=5, gamma=3) {

  eta <- mlflow_param("eta", eta, "numeric")
  max_depth <- mlflow_param("max_depth", max_depth, "numeric")
  gamma <- mlflow_param("gamma", gamma, "numeric")

  with(mlflow_start_run(nested=TRUE), {

    mlflow_log_param("eta", eta)
    mlflow_log_param("max_depth", max_depth)
    mlflow_log_param("gamma", gamma)

    num_class = length(levels(species))
    params = list(
      booster="gbtree",
      eta=eta,
      max_depth=max_depth,
      gamma=gamma,
      subsample=0.75,
      colsample_bytree=1,
      objective="multi:softprob",
      eval_metric="mlogloss",
      num_class=num_class
    )

    # Train the XGBoost classifer
    xgb_model=xgb.train(
      params=params,
      data=xgb_train,
      nrounds=10000,
      early_stopping_rounds=10,
      watchlist=list(val1=xgb_train,val2=xgb_test),
      verbose=0
    )

    # Predict outcomes with the test data
    xgb_pred = xgb_predict(xgb_model,test_data)

    # Calculate the final accuracy
    acc = 100*sum(xgb_pred$prediction==xgb_pred$label)/nrow(xgb_pred)
    #print(paste("Final Accuracy =",sprintf("%1.2f%%", acc)))

    message("Model accuracy: ", acc)
    mlflow_log_metric("accuracy", acc)

    mlflow_log_model(xgb_model, model_name, signature)

    # Log an attribute importance plot
    png(filename = "attribute_importance.png")
    importance_matrix <- xgb.importance(model = xgb_model)
    xgb.plot.importance(importance_matrix)
    dev.off()

    mlflow_log_artifact("attribute_importance.png")

  })

}

# COMMAND ----------

# Train three models using different hyperparameters
train_model(eta=0.001, max_depth=5, gamma=3)
train_model(eta=0.01, max_depth=10, gamma=3)
train_model(eta=0.0001, max_depth=15, gamma=3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Selecting the best performing model

# COMMAND ----------

runs <- mlflow_search_runs(order_by = "metrics.accuracy")

# Assuming the first run is the best model according to r^2
best_run_id <- runs$run_id[1]

# Construct model URI
model_uri <- paste0("runs:/", best_run_id, "/model")

# Construct model URI
message(c("run_id        : ", best_run_id))
message(c("experiment_id : ", runs$experiment_id[1]))

model_uri <- paste(runs$artifact_uri[1], model_name, sep = "/")
message(c("Model URI.    : ", model_uri, "\n"))

writeLines(as.character(best_run_id), "best_run_id.txt", useBytes = TRUE)

# COMMAND ----------

# MAGIC %python
# MAGIC import mlflow
# MAGIC
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC
# MAGIC catalog = dbutils.widgets.get("uc_catalog")
# MAGIC schema = dbutils.widgets.get("uc_schema")
# MAGIC model_name = dbutils.widgets.get("model_name")
# MAGIC
# MAGIC with open("best_run_id.txt", "r") as file:
# MAGIC     best_run_id = file.read().strip()
# MAGIC
# MAGIC if best_run_id:
# MAGIC   run_uri = f"runs:/{best_run_id}/{model_name}"
# MAGIC   mlflow.register_model(run_uri, f"{catalog}.{schema}.{model_name}")
# MAGIC else:
# MAGIC   print("No run ID set. Can't register model in UC.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline table

# COMMAND ----------

if (tolower(monitoring_mode) == "enabled") {
  message("Lakehouse Monitoring is in ENABLED mode. Creating the baseline table for Lakehouse Monitoring.")
} else {
  message("Lakehouse Monitoring is in DISABLED mode. Exit creating the basline table without blocking model deployment.")
  dbutils.notebook.exit(model_uri)
}



# COMMAND ----------

## Get the latest model version
model_version <- mlflow_get_latest_versions(model_name)
model_version_number <- tail(model_version, n=1)[[1]]$version

## Load the model
best_model <- mlflow_load_model(model_uri)
#best_model <- mlflow_get_model_version(model_name, model_version_number)
 
xgb_pred = xgb_predict(best_model,test_data)
xgb_pred <- xgb_pred %>% mutate(model_version = model_version_number)

## Take a look
display(xgb_pred)

# COMMAND ----------

sdf <- sparklyr::sdf_copy_to(sc, xgb_pred, overwrite = TRUE)

catalog <- dbutils.widgets.get("uc_catalog")
schema <- dbutils.widgets.get("uc_schema")
baseline_table <- dbutils.widgets.get("baseline_table")

output_table <- paste(catalog, schema, baseline_table, sep=".")

sparklyr::spark_write_table( 
  x = sdf, 
  name = output_table, 
  #mode = "overwrite"
  mode = "append"
) 


# COMMAND ----------

spark_disconnect(sc)
