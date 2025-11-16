# Databricks notebook source
# MAGIC %pip install -U -qqqq backoff databricks-langchain langgraph==0.5.3 uv databricks-agents mlflow-skinny[databricks] psycopg[binary,pool] databricks-sql-connector langgraph-checkpoint-postgres
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Testing Locally
from ..responses_agent_wrappers import AGENT
import mlflow
import uuid

# Set a custom experiment name
EXPERIMENT_NAME = "/Shared/multiagent_source_to_pay"
mlflow.set_experiment(EXPERIMENT_NAME)

thread_id = f"local-test-{str(uuid.uuid4())[:8]}"

result = AGENT.predict(
    {
        "input": [{"role": "user", "content": "Give me the details of PO PO1005"}],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print(result.model_dump(exclude_none=True))

# COMMAND ----------

result = AGENT.predict(
    {
        "input": [{"role": "user", "content": "Give me the available POs"}],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print(result.model_dump(exclude_none=True))

# COMMAND ----------

result = AGENT.predict(
    {
        "input": [{"role": "user", "content": "Give me the details of PO 789321"}],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print(result.model_dump(exclude_none=True))

# COMMAND ----------

result = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "How can talk to somebody to resolve this issue?",
            }
        ],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print(result.model_dump(exclude_none=True))

# COMMAND ----------

result = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "Yes",
            }
        ],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print(result.model_dump(exclude_none=True))

# COMMAND ----------

result = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "My Purchase Order is Missing. Please investigate it.",
            }
        ],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print(result.model_dump(exclude_none=True))

# COMMAND ----------

result = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "How can you help me?",
            }
        ],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print(result.model_dump(exclude_none=True))

# COMMAND ----------

# DBTITLE 1,Logged it as MlFLow Model
import mlflow
from pkg_resources import get_distribution

input_example = {
    "input": [{"role": "user", "content": "Give me the details of PO PO1000"}],
    "custom_inputs": {"thread_id": "test-session-1"},
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="AGENT",
        python_model="responses_agent_wrapper.py",
        input_example=input_example,
        code_paths=[
            "agents.py",
            "constants.py",
            "dataservice.py",
            "tools.py",
            "message_utils.py",
            "postgre_connection_manager.py",
        ],
        pip_requirements=[
            f"mlflow=={get_distribution('mlflow').version}",
            f"langchain-core=={get_distribution('langchain-core').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"psycopg[binary,pool]=={get_distribution('psycopg[binary,pool]').version}",
            f"databricks-sql-connector=={get_distribution('databricks-sql-connector').version}",
            f"langgraph-checkpoint-postgres=={get_distribution('langgraph-checkpoint-postgres').version}",
            f"psycopg=={get_distribution('psycopg').version}",
        ],
    )
print(f"Model logged under experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Testing Logged model
mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/AGENT",
    input_data={
        "input": [{"role": "user", "content": "Give me all the PO details"}],
        "custom_inputs": {"thread_id": "test-registry-1"},
    },
    env_manager="uv",
)


# COMMAND ----------

# DBTITLE 1,Registering the MLFLOW model
mlflow.set_registry_uri("databricks-uc")

# Catalog + schema + model name
catalog_name = "genesis_dev_platform"
schema_name = "playground"
model_name = "multiagent_source_to_pay"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"


# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# DBTITLE 1,Testing the Registered Model
# Test the registered model
import mlflow

model_uri = "models:/genesis_dev_platform.playground.multiagent_source_to_pay/42"
loaded_agent = mlflow.pyfunc.load_model(model_uri)

thread_id = f"local-mlflow-test-{str(uuid.uuid4())[:8]}"
# Test with the same payload that worked locally
result = loaded_agent.predict(
    {
        "input": [{"role": "user", "content": "Give me list of Invoices"}],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print("✅ Registry test result:", result)


# COMMAND ----------

# Test with the same payload that worked locally
result = loaded_agent.predict(
    {
        "input": [
            {"role": "user", "content": "What is the status of INV-2000 invoice"}
        ],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print("✅ Registry test result:", result)


# COMMAND ----------

# Test with the same payload that worked locally
result = loaded_agent.predict(
    {
        "input": [{"role": "user", "content": "When will this invoice be paid"}],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print("✅ Registry test result:", result)

# COMMAND ----------

# Test with the same payload that worked locally
result = loaded_agent.predict(
    {
        "input": [
            {"role": "user", "content": "What is the status of INV-2002 invoice"}
        ],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print("✅ Registry test result:", result)

# COMMAND ----------

# Test with the same payload that worked locally
result = loaded_agent.predict(
    {
        "input": [{"role": "user", "content": "Why was it rejected?"}],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print("✅ Registry test result:", result)

# COMMAND ----------

# Test with the same payload that worked locally
result = loaded_agent.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "How can I talk to somebody to resolve this issue?",
            }
        ],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print("✅ Registry test result:", result)

# COMMAND ----------

# Test with the same payload that worked locally
result = loaded_agent.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "Yes",
            }
        ],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print("✅ Registry test result:", result)

# COMMAND ----------

# Test with the same payload that worked locally
result = loaded_agent.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "This invoice is missing in the system. Please investigate the issue",
            }
        ],
        "custom_inputs": {"thread_id": thread_id},
    }
)

print("✅ Registry test result:", result)

# COMMAND ----------

# DBTITLE 1,Deploying the Model Serving Endpoint
from databricks import agents

# -----------------------------
# Step 1: Load secrets from Databricks Vault
# -----------------------------
DATABRICKS_HOST = "https://adb-1802336422986531.11.azuredatabricks.net"
DATABRICKS_TOKEN = dbutils.secrets.get(scope="playground_scope", key="DATABRICKS_TOKEN")


# -----------------------------
# Step 2: Environment variables for deployment
# -----------------------------
environment_vars = {
    "DATABRICKS_HOST": DATABRICKS_HOST,
    "DATABRICKS_TOKEN": DATABRICKS_TOKEN,
    "ENABLE_MLFLOW_TRACING": "true",
    "MLFLOW_EXPERIMENT_NAME": "/Shared/multiagent_source_to_pay",
    "MLFLOW_EXPERIMENT_ID": "2845496865144127",
}


# Catalog + schema + model name
catalog_name = "genesis_dev_platform"
schema_name = "playground"
model_name = "multiagent_source_to_pay"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

agents.deploy(
    UC_MODEL_NAME,
    42,
    # uc_registered_model_info.version,
    tags={"type": "blueprint", "owner": "Satish Gunisetty"},
    environment_vars=environment_vars,
)
