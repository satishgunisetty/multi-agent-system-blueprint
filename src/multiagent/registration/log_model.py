# Databricks notebook source
# MAGIC %pip install -U -qqqq backoff databricks-langchain langgraph==0.5.3 uv databricks-agents mlflow-skinny[databricks] psycopg[binary,pool] databricks-sql-connector langgraph-checkpoint-postgres
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Testing Locally
from src.multiagent.responses_agent_wrapper import AGENT
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

# Save as log_model_with_src.py (Databricks notebook cell)
import mlflow
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sys

# Set a custom experiment name
EXPERIMENT_NAME = "/Shared/multiagent_source_to_pay"
mlflow.set_experiment(EXPERIMENT_NAME)

# =========== CONFIG =============
REPO_ROOT = "/Workspace/Users/satish_gunisetty@epam.com/multi-agent-system-blueprint"
SRC_DIR = Path(REPO_ROOT) / "src"  # <--- we will bundle this directory
# =================================

# Ensure src is on sys.path so we can import AGENT
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the python_model object (must be mlflow.pyfunc.PythonModel-like or ResponsesAgent)
from src.multiagent.responses_agent_wrapper import AGENT

# Example input
input_example = {
    "input": [{"role": "user", "content": "Give me the details of PO PO1000"}],
    "custom_inputs": {"thread_id": "test-session-1"},
}


# helper to build pip requirements (optional)
def get_ver(pkg_name: str):
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return None


pip_requirements = []
for pkg in [
    "mlflow",
    "langchain-core",
    "databricks-langchain",
    "langgraph",
    "databricks-connect",
    "psycopg[binary,pool]",
    "databricks-sql-connector",
    "langgraph-checkpoint-postgres",
    "psycopg",
]:
    v = get_ver(pkg)
    pip_requirements.append(f"{pkg}=={v}" if v else pkg)

# ====== IMPORTANT: point code_paths to the SRC_DIR so artifact contains `src/...` ======
code_paths = [str(SRC_DIR)]

# sanity check
if not SRC_DIR.exists():
    raise FileNotFoundError(f"src directory not found at: {SRC_DIR}")

with mlflow.start_run() as run:
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="AGENT",
        python_model="../responses_agent_wrapper.py",  # object/class (not path string)
        code_paths=code_paths,  # bundles the whole src/ directory
        pip_requirements=pip_requirements,
        input_example=input_example,
    )

print("Logged model run_id:", run.info.run_id)


# COMMAND ----------

logged_agent_info.run_id

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
catalog_name = "agentic_ai_poc"
schema_name = "assets"
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
import uuid

model_uri = "models:/agentic_ai_poc.assets.multiagent_source_to_pay/2"
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

dbutils.secrets.listScopes()

# COMMAND ----------

# DBTITLE 1,Deploying the Model Serving Endpoint
from databricks import agents

# -----------------------------
# Step 1: Load secrets from Databricks Vault
# -----------------------------
DATABRICKS_HOST = "https://dbc-29254c33-0fad.cloud.databricks.com"
# DATABRICKS_TOKEN = dbutils.secrets.get(scope="sgun-scope", key="DATABRICKS_TOKEN")
secret_scope = "sgun-scope"


# -----------------------------
# Step 2: Environment variables for deployment
# -----------------------------
environment_vars = {
    "DATABRICKS_HOST": DATABRICKS_HOST,
    "DATABRICKS_TOKEN": f"{{{{secrets/{secret_scope}/DATABRICKS_TOKEN}}}}",
    "ENABLE_MLFLOW_TRACING": "true",
    "MLFLOW_EXPERIMENT_ID": "569153060676829",
}


# Catalog + schema + model name
catalog_name = "agentic_ai_poc"
schema_name = "assets"
model_name = "multiagent_source_to_pay"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=2,
    endpoint_name="source-to-pay",
    # uc_registered_model_info.version,
    tags={"type": "blueprint", "owner": "Satish Gunisetty"},
    environment_vars=environment_vars,
    scale_to_zero=True,
)
