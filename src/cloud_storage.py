from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()
storage_client = storage.Client()

def upload_blob(source_file_name, destination_blob_name, bucket_name="bts-ml-data"):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def download_blob(source_blob_name, destination_file_name, bucket_name="bts-ml-data"):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

def list_prev_k_blobs(data_dir, bucket_name="bts-ml-data", k=8):
    """
    Returns a list of the last k files inserted into the bucket
    that are in the folder data_dir. 
    """
    bucket_iter = storage_client.list_blobs(bucket_name, prefix=data_dir)
    blobs = [blob for blob in bucket_iter]
    return blobs[-k:]

def get_prev_k_blobs(data_dir, bucket_name="bts-ml-data", k=8):
    """
    Gets the k previous files from a specified Cloud Storage bucket
    which contains the folder "data_dir". data_dir must be specified
    so that the get operation does not retrieve every single file
    in the bucket. 
    """
    blobs_list = list_prev_k_blobs(str(data_dir), bucket_name, k)
    print(blobs_list)
    for blob in blobs_list:
        destination_file_name = data_dir / blob.name.split("/")[-1] # Download to local directory with same structure
        blob.download_to_filename(destination_file_name)

def check_gcloud_blob_exists(filename, bucket_name="bts-ml-data") -> bool:
    """
    Checks if a file with the name filename exists in the bucket
    with name bucket_name. 
    """
    bucket = storage_client.get_bucket(bucket_name)
    return bucket.blob(filename).exists(storage_client)


