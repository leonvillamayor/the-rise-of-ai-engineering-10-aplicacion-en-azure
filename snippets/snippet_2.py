# adapters/azure_blob.py
from io import BytesIO
import numpy as np
from azure.storage.blob import BlobServiceClient

class AzureBlobCentroidStore:
    def __init__(self, account_url: str, container: str, sas: str):
        self._container = (
            BlobServiceClient(account_url=account_url, credential=sas)
            .get_container_client(container)
        )

    def load(self, layer: str, date: str) -> np.ndarray:
        blob = self._container.get_blob_client(f"{layer}/{date}.npy")
        return np.load(BytesIO(blob.download_blob().readall()))

    def put(self, layer: str, date: str, vec: np.ndarray) -> None:
        buf = BytesIO(); np.save(buf, vec); buf.seek(0)
        self._container.upload_blob(
            f"{layer}/{date}.npy", buf.read(), overwrite=True
        )