from kfp.v2 import dsl
from kfp.v2.compiler import Compiler
from google.cloud import aiplatform
# Corrected import for Resource Manager
from google.cloud.resourcemanager_v3 import ProjectsClient
import argparse
import os # <-- Added missing import

# --- Constants ---
# Define your GCP project details
PROJECT_ID = "your-gcp-project-id"
REGION = "europe-west1"
BUCKET_NAME = "your-gcs-bucket-name"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline-root"
# This is a dynamic placeholder that will be replaced by the CI/CD pipeline
DOCKER_IMAGE_URI = "europe-west1-docker.pkg.dev/your-project/your-repo/your-image:latest"


# --- Pipeline Components ---

@dsl.component(
    base_image=DOCKER_IMAGE_URI,
)
def train_stacked_model(
    bucket: str,
    sarima_path: str,
    xgboost_path: str
):
    """This component runs the main training script."""
    import subprocess
    cmd = [
        "python", "train.py",
        "--bucket-name", bucket,
        "--sarima-path", sarima_path,
        "--xgboost-path", xgboost_path
    ]
    subprocess.run(cmd, check=True)


# --- NEW COMPONENT TO REGISTER THE MODEL ---
@dsl.component(
    packages_to_install=["google-cloud-aiplatform"],
)
def register_model(
    project: str,
    region: str,
    bucket: str,
    model_path: str, # The GCS path to the XGBoost model
    model_display_name: str,
):
    """Uploads the trained model from GCS to the Vertex AI Model Registry."""
    from google.cloud import aiplatform
    import os # <-- Added missing import here as well for component context

    aiplatform.init(project=project, location=region, staging_bucket=bucket)

    # This is the serving container image for XGBoost models on Vertex AI
    serving_image = "europe-west1-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1.6:latest"

    # Check if a model with this name already exists to handle versioning
    existing_models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        location=region,
    )

    parent_model = None
    if existing_models:
        parent_model = existing_models[0].resource_name
        print(f"Found existing model: {parent_model}. Registering a new version.")
    else:
        print("No existing model found. Registering a new model.")

    # Upload the model to the registry
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=f"gs://{bucket}/{os.path.dirname(model_path)}", # The GCS directory
        serving_container_image_uri=serving_image,
        parent_model=parent_model, # This ensures versioning
        is_default_version=True,
    )

    print(f"Model registered. Resource name: {model.resource_name}")


@dsl.pipeline(
    name="demand-forecasting-pipeline",
    description="A pipeline to train a stacked SARIMA+XGBoost model and register it.",
    pipeline_root=PIPELINE_ROOT,
)
def forecasting_pipeline(
    project_id: str = PROJECT_ID,
    bucket_name: str = BUCKET_NAME,
    region: str = REGION
):
    # Define the GCS paths for the models
    sarima_gcs_path = "models/sarima/sarima_model.pkl"
    xgboost_gcs_path = "models/xgboost/xgboost_model.pkl"

    train_op = train_stacked_model(
        bucket=bucket_name,
        sarima_path=sarima_gcs_path,
        xgboost_path=xgboost_gcs_path,
    )

    # --- NEW STEP TO REGISTER THE MODEL ---
    # This step runs only after the training step is successful
    register_op = register_model(
        project=project_id,
        region=region,
        bucket=bucket_name,
        model_path=xgboost_gcs_path, # We are registering the XGBoost model
        model_display_name="demand-forecaster-stacked"
    ).after(train_op)


# --- Main execution block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Compile the pipeline to a JSON file.'
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run the pipeline on Vertex AI.'
    )
    args = parser.parse_args()

    if args.compile or not args.run:
        Compiler().compile(
            pipeline_func=forecasting_pipeline,
            package_path="forecasting_pipeline.json"
        )
        print("Pipeline compiled to forecasting_pipeline.json")

    if args.run:
        print("Submitting pipeline job to Vertex AI...")
        aiplatform.init(project=PROJECT_ID, location=REGION)

        # Get project number to construct service account email
        # Correctly instantiate the client
        client = ProjectsClient()
        project_details = client.get_project(name=f"projects/{PROJECT_ID}")
        project_number = project_details.project_number
        service_account = f"{project_number}-compute@developer.gserviceaccount.com"

        job = aiplatform.PipelineJob(
            display_name="demand-forecasting-pipeline-run",
            template_path="forecasting_pipeline.json",
            pipeline_root=PIPELINE_ROOT,
            parameter_values={
                "project_id": PROJECT_ID,
                "bucket_name": BUCKET_NAME,
                "region": REGION
            },
        )
        
        job.run(service_account=service_account)
        print("Pipeline job started.")

