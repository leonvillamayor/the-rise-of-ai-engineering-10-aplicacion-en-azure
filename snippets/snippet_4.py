# episodio 10 · «Aplicación en Azure» — adapter CentroidStore
# la política (válida(c), alerta_t, τ, versionado) viaja, no el código
from __future__ import annotations

import io
import os
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. CONTRATO (la parte que viaja) — cerrado, no se toca al cambiar proveedor
# ──────────────────────────────────────────────────────────────────────────────
class CentroidStore(Protocol):
    """Lo que DefensiveOutlierPipeline consume por inyección de dependencias.

    El pipeline NO ve boto3 ni azure-storage-blob; solo este contrato.
    Mismo layout en ambos proveedores: <prefix>/<layer>/<YYYY-MM-DD>.npy
    """

    def load(self, layer: str, on: date) -> np.ndarray: ...
    def latest(self, layer: str) -> np.ndarray: ...


# ──────────────────────────────────────────────────────────────────────────────
# 2. IMPLEMENTACIÓN AWS — ya en producción desde episodios 8-9
# ──────────────────────────────────────────────────────────────────────────────
class S3CentroidStore:
    """Cumple CentroidStore sobre s3://autodoc-centroids/<layer>/<date>.npy"""

    def __init__(self, bucket: str, prefix: str = "autodoc-centroids",
                 client=None) -> None:
        import boto3  # import diferido: evita coste en cold-start de Azure
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._s3 = client or boto3.client("s3")

    def _key(self, layer: str, on: date) -> str:
        return f"{self._prefix}/{layer}/{on.isoformat()}.npy"

    def load(self, layer: str, on: date) -> np.ndarray:
        buf = io.BytesIO()
        self._s3.download_fileobj(self._bucket, self._key(layer, on), buf)
        buf.seek(0)
        return _validate_centroid(np.load(buf, allow_pickle=False))

    def latest(self, layer: str) -> np.ndarray:
        paginator = self._s3.get_paginator("list_objects_v2")
        dated: list[tuple[date, str]] = []
        for page in paginator.paginate(
            Bucket=self._bucket, Prefix=f"{self._prefix}/{layer}/"
        ):
            for obj in page.get("Contents", []):
                name = Path(obj["Key"]).name  # YYYY-MM-DD.npy
                if len(name) == 15 and name.endswith(".npy"):
                    dated.append((date.fromisoformat(name[:10]), obj["Key"]))
        if not dated:
            raise FileNotFoundError(f"sin centroides para layer={layer!r}")
        dated.sort()
        return self.load(layer, dated[-1][0])


# ──────────────────────────────────────────────────────────────────────────────
# 3. IMPLEMENTACIÓN AZURE — novedad del episodio 10
# ──────────────────────────────────────────────────────────────────────────────
class BlobCentroidStore:
    """Cumple CentroidStore sobre azure://<container>/<prefix>/<layer>/<date>.npy

    Replica exactamente la jerarquía de S3 para que las migraciones sean
    un copia-y-pega (sin reescribir el indexer de la Lambda → Function).
    """

    def __init__(self, account_url: str, container: str,
                 prefix: str = "autodoc-centroids", credential=None) -> None:
        from azure.storage.blob import BlobServiceClient  # import diferido
        from azure.identity import DefaultAzureCredential
        self._container = container
        self._prefix = prefix.rstrip("/")
        cred = credential or DefaultAzureCredential()
        self._svc = BlobServiceClient(account_url=account_url, credential=cred)

    def _blob(self, layer: str, on: date) -> str:
        return f"{self._prefix}/{layer}/{on.isoformat()}.npy"

    def load(self, layer: str, on: date) -> np.ndarray:
        client = self._svc.get_blob_client(
            container=self._container, blob=self._blob(layer, on)
        )
        data = client.download_blob().readall()
        return _validate_centroid(np.load(io.BytesIO(data), allow_pickle=False))

    def latest(self, layer: str) -> np.ndarray:
        container = self._svc.get_container_client(self._container)
        dated: list[tuple[date, str]] = []
        for blob in container.list_blobs(
            name_starts_with=f"{self._prefix}/{layer}/"
        ):
            name = Path(blob.name).name
            if len(name) == 15 and name.endswith(".npy"):
                dated.append((date.fromisoformat(name[:10]), blob.name))
        if not dated:
            raise FileNotFoundError(f"sin centroides para layer={layer!r}")
        dated.sort()
        most_recent_layer, most_recent_date = dated[-1]
        return self.load(most_recent_layer, most_recent_date)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Validación compartida — la fuente auditable, no el byte-path
# ──────────────────────────────────────────────────────────────────────────────
def _validate_centroid(arr: np.ndarray) -> np.ndarray:
    """Defensa común: si el .npy no es un vector 1024-d, no es c̄."""
    if arr.ndim != 1 or arr.shape[0] != 1024:
        raise ValueError(
            f"centroide con forma inválida {arr.shape}; "
            "Bedrock Titan v2 exige dim=1024 y vector 1-D"
        )
    return arr.astype(np.float32, copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# 5. CONSUMIDOR — DefensiveOutlierPipeline NO cambia entre AWS y Azure
# ──────────────────────────────────────────────────────────────────────────────
class DefensiveOutlierPipeline:
    def __init__(self, store: CentroidStore, embedder,
                 drop_mode: str = "quarantine",
                 tau: float = 0.10) -> None:  # τ interno (ep. 8)
        self._store = store        # <- DI: solo el contrato
        self._embedder = embedder
        self._drop_mode = drop_mode
        self._tau = tau

    def evaluate(self, chunks: list[str], layer: str) -> list[dict]:
        c_bar = self._store.latest(layer)  # c̄ desde donde sea
        out: list[dict] = []
        for text in chunks:
            v = self._embedder.embed(text)
            d_i = float(
                1.0 - np.dot(v, c_bar) /
                (np.linalg.norm(v) * np.linalg.norm(c_bar) + 1e-12)
            )
            out.append({
                "text": text,
                "d_i": d_i,
                "alerta_t": int(d_i > self._tau),
                "drop_mode": self._drop_mode,
            })
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 6. COMPOSITION ROOT — único punto donde se elige proveedor
# ──────────────────────────────────────────────────────────────────────────────
def build_pipeline(env: str | None = None) -> DefensiveOutlierPipeline:
    """Selecciona el backend sin que la pipeline se entere.

    En Azure Functions (ep. 10) este factory vive en function_app.py
    y se invoca una sola vez por instancia (singleton del host).
    """
    from autodoc.embedders import BedrockEmbedder  # contrato embed() intacto

    env = env or os.getenv("AUTODOC_ENV", "aws")
    embedder = BedrockEmbedder(model_id="amazon.titan-embed-text-v2:0",
                               dims=1024)

    if env == "aws":
        store: CentroidStore = S3CentroidStore(
            bucket=os.environ["CENTROIDS_BUCKET"],
        )
    elif env == "azure":
        store = BlobCentroidStore(
            account_url=os.environ["AZURE_BLOB_ACCOUNT_URL"],
            container=os.environ["AZURE_BLOB_CONTAINER"],
        )
    else:
        raise ValueError(f"AUTODOC_ENV desconocido: {env!r}")

    return DefensiveOutlierPipeline(store=store, embedder=embedder)


# ──────────────────────────────────────────────────────────────────────────────
# 7. SMOKE TEST — misma demo de 8 chunks del ep. 8, ahora sobre Azure
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.environ.setdefault("AUTODOC_ENV", "azure")
    os.environ.setdefault("AZURE_BLOB_ACCOUNT_URL",
                          "https://autodocprod.blob.core.windows.net")
    os.environ.setdefault("AZURE_BLOB_CONTAINER", "autodoc-centroids")

    pipeline = build_pipeline()
    sample = [
        "El plan cuesta 28 EUR/día",          # válido
        "tortilla de patatas con cebolla",    # ruido-2 (semántico)
        "#TODO refactor del indexer",         # ruido-3 (regex)
        "Cobertura médica completa incluida", # válido
        "asdf qwer zxcv",                     # ruido-1 (MAD z=4.2)
        "Cancelación sin coste durante 14 días",
        "ruiidodd-99 sin sentido",            # ruido-1
        "Soporte 24/7 por chat y email",      # válido
    ]
    report = pipeline.evaluate(sample, layer="pricing-es")
    for r in report:
        print(f"d_i={r['d_i']:.3f}  alerta_t={r['alerta_t']}  "
              f"action={r['drop_mode']}  ← {r['text'][:40]}")