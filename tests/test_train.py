# tests/test_train.py

import os
from unittest.mock import patch
from src.train import train_and_save_models

# This decorator intercepts calls to the GCS client, allowing us to
# test the script's logic without needing a real cloud connection.
@patch('src.train.storage.Client')
def test_train_script_runs_and_creates_artifacts(mock_storage_client, tmpdir):
    """
    A smoke test to ensure the training function runs without errors
    and saves the two expected model files.
    """
    # pytest provides a temporary directory (tmpdir) for the test
    temp_sarima_path = os.path.join(tmpdir, 'sarima_model.pkl')
    temp_xgboost_path = os.path.join(tmpdir, 'xgboost_model.pkl')
    
    # Run the main training function, telling it to use our temporary paths
    train_and_save_models(
        bucket_name='mock-bucket',
        sarima_path=temp_sarima_path,
        xgboost_path=temp_xgboost_path
    )
    
    # Assert that the two model files were created in the temporary directory
    assert os.path.exists(temp_sarima_path)
    assert os.path.exists(temp_xgboost_path)
    
    print("\nSmoke test passed: Training script ran and created model artifacts.")