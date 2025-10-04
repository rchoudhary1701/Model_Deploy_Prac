# src/pipeline.py
from kfp.v2 import dsl
from kfp.v2.compiler import Compiler
from google.cloud import aiplatform

# Define your GCP project details
PROJECT_ID = "your-gcp-project-id"
REGION = "europe-west1"
BUCKET_NAME = "your-gcs-bucket-name"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline-root"
DOCKER_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/forecasting-repo/demand-forecaster:latest"

@dsl.component(
    base_image=DOCKER_IMAGE_URI,
    packages_to_install=["google-cloud-storage", "joblib", "pandas", "numpy", "xgboost", "statsmodels"]
)
def train_stacked_model(
    project: str,
    bucket: str,
    sarima_path: str,
    xgboost_path: str
):
    # This component simply runs our train.py script
    # The base_image already contains all the logic
    # We are just passing arguments to it
    import subprocess
    cmd = [
        "python", "train.py",
        "--bucket-name", bucket,
        "--sarima-path", sarima_path,
        "--xgboost-path", xgboost_path
    ]
    subprocess.run(cmd, check=True)

@dsl.pipeline(
    name="demand-forecasting-pipeline",
    description="A pipeline to train a stacked SARIMA+XGBoost model.",
    pipeline_root=PIPELINE_ROOT,
)

def forecasting_pipeline(
    project_id: str = PROJECT_ID,
    bucket_name: str = BUCKET_NAME
):
    train_op = train_stacked_model(
        project=project_id,
        bucket=bucket_name,
        sarima_path="models/sarima/sarima_model.pkl",
        xgboost_path="models/xgboost/xgboost_model.pkl"
    )

# --- Compile and run the pipeline ---
if __name__ == '__main__':
    Compiler().compile(
        pipeline_func=forecasting_pipeline,
        package_path="forecasting_pipeline.json"
    )

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.PipelineJob(
        display_name="demand-forecasting-pipeline-run",
        template_path="forecasting_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project_id": PROJECT_ID,
            "bucket_name": BUCKET_NAME
        }
    )
    print("Starting Vertex AI Pipeline job...")
    job.run()
    print("Job started.")
