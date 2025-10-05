import argparse
from kfp.v2 import dsl
from kfp.v2.compiler import Compiler
from google.cloud import aiplatform

# This DOCKER_IMAGE_URI will be dynamically replaced by the Cloud Build CI/CD pipeline
DOCKER_IMAGE_URI = "placeholder-image-uri" 

# Define a pipeline component for training
@dsl.component(
    base_image=DOCKER_IMAGE_URI,
)
def train_stacked_model(
    project: str,
    bucket: str,
    sarima_path: str,
    xgboost_path: str
):
    """
    This component runs the main training script.
    The Docker container already has all the logic.
    """
    import subprocess
    cmd = [
        "python", "train.py",
        "--bucket-name", bucket,
        "--sarima-path", sarima_path,
        "--xgboost-path", xgboost_path
    ]
    subprocess.run(cmd, check=True)

# Define the main pipeline structure
@dsl.pipeline(
    name="demand-forecasting-pipeline",
    description="A pipeline to train a stacked SARIMA+XGBoost model.",
)
def forecasting_pipeline(
    project_id: str,
    bucket_name: str,
    region: str = "europe-west1"
):
    train_op = train_stacked_model(
        project=project_id,
        bucket=bucket_name,
        sarima_path="models/sarima/sarima_model.pkl",
        xgboost_path="models/xgboost/xgboost_model.pkl"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, required=True, help="Your GCP project ID.")
    parser.add_argument("--bucket_name", type=str, required=True, help="Your GCS bucket name.")
    parser.add_argument("--run", action="store_true", help="Set this flag to submit the pipeline job.")
    args = parser.parse_args()

    COMPILED_PIPELINE_PATH = "forecasting_pipeline.json"
    
    print("Compiling pipeline...")
    Compiler().compile(
        pipeline_func=forecasting_pipeline,
        package_path=COMPILED_PIPELINE_PATH
    )
    print(f"Pipeline compiled to {COMPILED_PIPELINE_PATH}")

    if args.run:
        print("Submitting pipeline job to Vertex AI...")
        aiplatform.init(project=args.project_id, location="europe-west1")

        # --- THE FIX IS HERE ---
        # Get the project number to construct the service account email
        project_number = aiplatform.initializer.global_config.project_number
        SERVICE_ACCOUNT = f"{project_number}-compute@developer.gserviceaccount.com"
        print(f"Using Service Account: {SERVICE_ACCOUNT}")

        job = aiplatform.PipelineJob(
            display_name="demand-forecasting-pipeline-run",
            template_path=COMPILED_PIPELINE_PATH,
            pipeline_root=f"gs://{args.bucket_name}/pipeline-root",
            parameter_values={
                "project_id": args.project_id,
                "bucket_name": args.bucket_name
            }
        )
        
        # We now explicitly tell the job to run with our configured service account
        job.run(service_account=SERVICE_ACCOUNT)
        print("Pipeline job submitted successfully.")
    else:
        print("Compilation complete. To run the pipeline, add the --run flag.")

