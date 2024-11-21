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
packages <- c("dplyr", "VGAM", "sparklyr", "mlflow", "carrier")

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

iris_df$Species <- as.numeric(as.factor(iris_df$Species))

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

train_data = iris_df[train_ind,]
test_data = iris_df[-train_ind,]

display(train_data)

# COMMAND ----------

cat("Number of samples in the training set:", nrow(train_data), "\n")
cat("Number of samples in the test set.   :", nrow(test_data))

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

sapply(train_data, typeof)


# COMMAND ----------

extract_schema <- function(df) {
  schema <- lapply(names(df), function(col_name) {
    list(type = typeof(df[[col_name]])[1], name = col_name)
  })
  return(schema)
}


input_schema <- extract_schema(iris_df[, names(iris_df) != "Species"] )
signature <- list(
    inputs = input_schema,
    outputs = list(list(type = "double")))

input_schema

# COMMAND ----------



# COMMAND ----------




# COMMAND ----------

mlflow_set_experiment(experiment_name)

train_model <- function(...) {

  sample_weighting <- mlflow_param("sample_weighting", "inverse_number_of_samples", "string")
  
  with(mlflow_start_run(nested=TRUE), {
    model <- vglm(Species~., data=iris_df, family=multinomial, sample_weighting=inverse_number_of_samples)
    
    mlflow_log_param("sample_weighting", sample_weighting)

    crate_model <- crate(
      function(new_obs)  stats::predict(model, data = new_obs),
      model = model
    )

    mlflow_log_metric("ResSS", model@ResSS)
    mlflow_log_metric("AIC", AIC(model))

    mlflow_log_model(crate_model, model_name, signature)
  }
  )


}

# COMMAND ----------

train_model()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registering model

# COMMAND ----------

# MAGIC %python
# MAGIC import mlflow
# MAGIC
# MAGIC best_run_id="ee70e4c934954d24a533b194a1d81469"
# MAGIC
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC
# MAGIC catalog = dbutils.widgets.get("uc_catalog")
# MAGIC schema = dbutils.widgets.get("uc_schema")
# MAGIC model_name = dbutils.widgets.get("model_name")
# MAGIC
# MAGIC #with open("best_run_id.txt", "r") as file:
# MAGIC #    best_run_id = file.read().strip()
# MAGIC
# MAGIC if best_run_id:
# MAGIC   run_uri = f"runs:/{best_run_id}/{model_name}"
# MAGIC   mlflow.register_model(run_uri, f"{catalog}.{schema}.{model_name}")
# MAGIC else:
# MAGIC   print("No run ID set. Can't register model in UC.")
