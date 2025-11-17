# Databricks notebook source
# MAGIC %pip install -U -qqqq backoff databricks-langchain langgraph==0.5.3 uv databricks-agents mlflow-skinny[databricks] psycopg[binary,pool] databricks-sql-connector langgraph-checkpoint-postgres
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

EXPERIMENT_NAME = "/Shared/evaluation_multiagent_source_to_pay"
mlflow.set_experiment(EXPERIMENT_NAME)

# Fixed eval_data - the key "inputs_dict" must match your predict_fn parameter name
eval_data = [
    {
        "inputs": {
            "inputs_dict": {  # This key must match your predict_fn parameter
                "input": [
                    {
                        "role": "user",
                        "content": "What is the status of PO 10000?",
                        "type": "message"
                    }
                ],
                "custom_inputs": {
                    "thread_id": "eval-session-1"
                }
            }
        },
        "expectations": {
            "expected_response": "Purchase Order 10000 is not available..."
        },
    },
    {
        "inputs": {
            "inputs_dict": {
                "input": [
                    {
                        "role": "user", 
                        "content": "Give me all pending purchase orders",
                        "type": "message"
                    }
                ],
                "custom_inputs": {
                    "thread_id": "eval-session-2"
                }
            }
        },
        "expectations": {
            "expected_response": "List or table of pending POs."
        },
    },
    {
        "inputs": {
            "inputs_dict": {
                "input": [
                    {
                        "role": "user",
                        "content": "Show me the rejected invoices",
                        "type": "message"
                    }
                ],
                "custom_inputs": {
                    "thread_id": "eval-session-3"
                }
            }
        },
        "expectations": {
            "expected_response": "List or table of rejected invoices."
        },
    },
    {
        "inputs": {
            "inputs_dict": {
                "input": [
                    {
                        "role": "user",
                        "content": "Hello",
                        "type": "message"
                    }
                ],
                "custom_inputs": {
                    "thread_id": "eval-session-5"
                }
            }
        },
        "expectations": {
            "expected_response": "A greeting and then offers assistance with specific services (POs, invoices, or ServiceNow tickets)"
        },
    },
]

eval_df = pd.DataFrame(eval_data)

model_name = "agentic_ai_poc.assets.multiagent_source_to_pay"
model_version = 2

loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

def predict_fn(inputs_dict):
    """
    The parameter name 'inputs_dict' must match the key in your eval_data inputs.
    """
    try:
        print(f"📝 Input received: {inputs_dict}")
        
        # Call the model with the correct format
        results = loaded_model.predict(inputs_dict)
        
        print(f"📤 Model returned: {type(results)}")
        
        # Extract the text content from your ResponsesAgent output format
        if isinstance(results, dict):
            # Handle ResponsesAgent format: {'object': 'response', 'output': [...]}
            if 'output' in results and isinstance(results['output'], list):
                output_item = results['output'][0]
                if 'content' in output_item and isinstance(output_item['content'], list):
                    # Extract text from content array
                    text_content = output_item['content'][0].get('text', '')
                    print(f"✅ Extracted text: {text_content[:100]}...")
                    return text_content
                elif 'content' in output_item:
                    return str(output_item['content'])
            # Fallback for other dict formats
            elif 'content' in results:
                return results['content']
        
        # Handle list format
        elif isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], dict) and 'content' in results[0]:
                return results[0]['content']
        
        # Fallback to string conversion
        return str(results)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return f"Error: {str(e)}"

# Test the prediction function first
print("🧪 Testing prediction function...")
test_input = eval_data[0]["inputs"]["inputs_dict"]
test_result = predict_fn(test_input)
print(f"Test result: {test_result}")

# Run evaluation
print("🚀 Starting evaluation...")
results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=predict_fn,
    scorers=[
        Correctness(),
        Safety(),
        RelevanceToQuery(),
        Guidelines(
            guidelines="""Response must be professional"""
        ),
    ],
)

print("✅ Evaluation completed!")
results

