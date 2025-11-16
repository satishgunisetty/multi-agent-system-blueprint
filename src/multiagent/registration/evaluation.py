# Databricks notebook source
# MAGIC %pip install -U -qqqq backoff databricks-langchain langgraph==0.5.3 uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import (
    Guidelines,
    Safety,
    Correctness,
    RelevanceToQuery,
    RetrievalGroundedness,
)
import pandas as pd

eval_data = [
    {
        "inputs": {
            "batch_inputs": [
                {
                    "messages": "What is the status of PO 12345?",
                    "session_id": "eval-session",
                }
            ]
        },
        "expectations": {
            "expected_response": "Here is the information regarding the purchase order: ..."
        },
    },
    {
        "inputs": {
            "batch_inputs": [
                {
                    "messages": "Update the quantity for PO 23456 to 20.",
                    "session_id": "eval-session",
                }
            ]
        },
        "expectations": {
            "expected_response": "Could not update PO 23456. There is no PO 23456"
        },
    },
    # Add more rows as needed
]
eval_df = pd.DataFrame(eval_data)

model_name = "genesis_dev_platform.playground.multiagent_source_to_pay"
model_version = 8

loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")


def predict_fn(batch_inputs):
    # Always return a plain Python list (not pandas Series/DataFrame)
    results = loaded_model.predict(batch_inputs)
    if isinstance(results, pd.Series):
        return results.tolist()
    elif isinstance(results, pd.DataFrame):
        return results.squeeze().tolist()
    elif not isinstance(results, list):
        return [str(results)]
    return results


results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=predict_fn,
    scorers=[
        Correctness(),
        Safety(),
        Guidelines(
            guidelines="Response must be professional and helpful",
            name="professional_tone",
        ),
        RelevanceToQuery(),
        RetrievalGroundedness(),
    ],
)

results


# COMMAND ----------

# DBTITLE 1,Deploy
from databricks import agents

uc_model_name = "genesis_dev_platform.playground.multiagent_source_to_pay"
uc_model_version = 8  # or "latest"

deployment = agents.deploy(uc_model_name, uc_model_version)

print("Serving endpoint URL:", deployment.query_endpoint)
