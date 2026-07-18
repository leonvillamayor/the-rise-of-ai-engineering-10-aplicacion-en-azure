# =============================================================================
#  The Rise of AI Engineering — Ep.10 · Aplicación en Azure
#  Indexer como Azure Function: trigger Blob + DriftReport en App Insights
# =============================================================================
#  Recap (cierre ep.9 → apertura ep.10):
#  En AWS demostramos que `DefensiveOutlierPipeline` y `CentroidStore` son
#  un *patrón portable*: la política (válida(c), alerta_t, τ, versionado)
#  viaja, no el código. Aquí ejecutamos exactamente esa misma pipeline
#  sobre Azure Blob, Azure Functions y Application Insights — el adapter
#  del capítulo anterior se cierra y el Function se reduce al trigger.
# =============================================================================

from __future__ import annotations

import os
import json
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence

import numpy as np
import azure.functions as func   # azure-functions >= 1.20
from azure.storage.blob import (
    BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions,
)
from azure.eventgrid import EventGridPublisherClient
from azure.core.credentials import AzureKeyCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace, metrics

# ──────────────────────────────────────────────────────────────────────────────
# 1. Lo YA ENSEÑADO (ep. 1–9) — se reusa TAL CUAL desde `autodoc_core`.
#    NO se redefine: contrato estable, solo cambia el adapter debajo.
# ──────────────────────────────────────────────────────────────────────────────
# from autodoc_core import (                        # paquete del curso
#     DefensiveOutlierPipeline,                    # drop_mode="quarantine"
#     DriftReport,                                 # d_i = 1 - cos(c_i, c̄)
#     CentroidStore,                               # Protocol: get/put
#     BedrockEmbedder,                             # contrato embed(text)->Vector
#     alerta_t, valida_citacion,                   # τ finanzas / interno
# )
#
# Para mantener el ejemplo autocontenido (y que el artículo compile
# sin dependencias privadas), re-declaramos las firmas mínimas que el
# Function necesita. En el repo del curso se importan del paquete.

logger = logging.getLogger("autodoc.indexer")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Contratos portables (idénticos a los del ep.9, solo cambia el adapter)
# ──────────────────────────────────────────────────────────────────────────────
class CentroidStore(Protocol):
    """Contrato portable: ayer S3, hoy Azure Blob, mañana GCS."""
    def get(self, layer: str, date: str) -> np.ndarray: ...
    def put(self, layer: str, date: str, vector: np.ndarray) -> None: ...

class Embedder(Protocol):
    """Contrato estable `embed(text)->Vector`. Bedrock ayer, Foundry hoy."""
    def embed(self, text: str) -> np.ndarray: ...

@dataclass(frozen=True)
class DriftReport:
    chunk_hash: str
    layer: str
    length_chars: int
    shannon_entropy: float
    cosine_to_centroid: float       # d_i = 1 - cos(c_i, c̄)
    alerta_t: int                   # 𝟙[d_i > τ]
    citacion_valida: bool
    quarantine: bool
    rule_id: str                    # reglaset versionado


# ──────────────────────────────────────────────────────────────────────────────
# 3. Adapter Azure para CentroidStore  (mirror del S3 adapter del ep.9)
# ──────────────────────────────────────────────────────────────────────────────
class AzureBlobCentroidStore:
    """Implementa CentroidStore sobre `azure://autodoc-centroids/...` .

    El interfaz es idéntico al del ep.9 (S3): `c̄` vive en
        azure://autodoc-centroids/<layer>/<date>.npy
    y la pipeline NO se entera del cambio de proveedor.
    """

    def __init__(self, conn_str: str, container: str = "autodoc-centroids") -> None:
        self._svc = BlobServiceClient.from_connection_string(conn_str)
        self._container = container

    def _blob_name(self, layer: str, date: str) -> str:
        return f"{layer}/{date}.npy"

    def get(self, layer: str, date: str) -> np.ndarray:
        blob = self._svc.get_blob_client(self._container, self._blob_name(layer, date))
        return np.load(BytesIO(blob.download_blob().readall()))

    def put(self, layer: str, date: str, vector: np.ndarray) -> None:
        blob = self._svc.get_blob_client(self._container, self._blob_name(layer, date))
        blob.upload_blob(
            np.save(BytesIO(), vector, allow_pickle=False),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/octet-stream"),
        )

# ──────────────────────────────────────────────────────────────────────────────
# 4. Embedder Azure AI Foundry  (sustituye a BedrockEmbedder del ep.9)
#    Mismo contrato `embed(text)->Vector` → 1024-d, drop-in replacement.
# ──────────────────────────────────────────────────────────────────────────────
class FoundryEmbedder:
    """Adapter para un modelo de embeddings desplegado en Azure AI Foundry.

    En el curso desplegamos Cohere embed-v3 o text-embedding-3-large (1024-d)
    en un proyecto Foundry; el contrato `embed(text)->Vector` se preserva.
    """

    def __init__(self, endpoint: str, api_key: str, deployment: str, dim: int = 1024):
        from azure.ai.inference import EmbeddingsClient
        self._client = EmbeddingsClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )
        self._deployment = deployment
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        resp = self._client.embed(
            input=[text], model=self._deployment
        )
        return np.asarray(resp.data[0].embedding, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Métricas → Application Insights  (mirror de CloudWatch en ep.9)
# ──────────────────────────────────────────────────────────────────────────────
configure_azure_monitor(connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"])
tracer = trace.get_tracer("autodoc.indexer")
meter = metrics.get_meter("autodoc.indexer")

m_quarantine_rate = meter.create_observable_gauge(
    "quarantine_rate_24h",
    callbacks=[lambda opts: (_compute_quarantine_rate_24h(),)],
    description="Tasa de cuarentena rolling-window 24h (mirror CloudWatch)",
)
m_length = meter.create_histogram("length_chars")
m_entropy = meter.create_histogram("shannon_entropy")
m_cos = meter.create_histogram("cosine_to_centroid")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Function App — el indexer. Se reduce al trigger.
# ──────────────────────────────────────────────────────────────────────────────
app = func.FunctionApp()

# Singleton por proceso: conexiones pesadas se crean una vez.
_CENTROIDS = AzureBlobCentroidStore(os.environ["AZURE_BLOB_CONNECTION_STRING"])
_EMBEDDER = FoundryEmbedder(
    endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"],
    api_key=os.environ["AZURE_FOUNDRY_KEY"],
    deployment=os.environ["AZURE_FOUNDRY_DEPLOYMENT"],
)
_PIPELINE = None  # DefensiveOutlierPipeline del paquete autodoc_core
_EVENTGRID = EventGridPublisherClient(
    os.environ["EVENT_GRID_TOPIC_ENDPOINT"], AzureKeyCredential(os.environ["EVENT_GRID_TOPIC_KEY"])
)

# In-memory rolling window de 24h (en producción → Cosmos DB o Redis;
# el contrato es el mismo que la DynamoDB rolling window del ep.9).
_QUARANTINE_TS: list[datetime] = []


def _compute_quarantine_rate_24h(now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    cutoff = now.timestamp() - 24 * 3600
    _QUARANTINE_TS[:] = [t for t in _QUARANTINE_TS if t.timestamp() >= cutoff]
    return len(_QUARANTINE_TS) / max(1, _WINDOW_TOTAL_24H)


@app.blob_trigger(arg_name="chunk", path="autodoc-chunks/{name}",
                  connection="AZURE_BLOB_CONNECTION_STRING")
def indexer(chunk: func.InputStream) -> None:
    """Trigger Blob: un nuevo chunk en `autodoc-chunks/` → ejecuta la pipeline.

    Mismo contrato que el Lambda indexer del ep.9, misma política:
        evalúa en index-time (NO en query-time), escribe DriftReport,
        y publica en Event Grid para que Logic Apps / Pipelines reaccionen.
    """
    with tracer.start_as_current_span("indexer.process_chunk") as span:
        text = chunk.read().decode("utf-8", errors="replace")
        layer = _layer_from_blob_name(chunk.name)
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # 1) c̄ desde el adapter Azure (idéntico a S3 en ep.9)
        c_bar = _CENTROIDS.get(layer, date)

        # 2) embedding — mismo contrato `embed(text)->Vector`
        c_i = _EMBEDDER.embed(text)

        # 3) DriftReport (eval, no juicio): d_i = 1 - cos(c_i, c̄)
        cos = float(np.dot(c_i, c_bar) / (np.linalg.norm(c_i) * np.linalg.norm(c_bar) + 1e-12))
        d_i = 1.0 - cos

        # 4) alerta_t (umbral τ por capa: 0.05 finanzas, 0.10 interno)
        tau = {"finanzas": 0.05, "interno": 0.10}.get(layer, 0.10)
        alerta_t = int(d_i > tau)

        # 5) cita válida(c) — misma regla regex/lookup del ep.9
        citacion_valida = _valida_citacion(text, layer)

        # 6) quarantine si outlier o cita inválida
        quarantine = bool(alerta_t or not citacion_valida)

        report = DriftReport(
            chunk_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            layer=layer,
            length_chars=len(text),
            shannon_entropy=_shannon_entropy(text),
            cosine_to_centroid=cos,
            alerta_t=alerta_t,
            citacion_valida=citacion_valida,
            quarantine=quarantine,
            rule_id=os.environ.get("AUTODOC_RULESET_VERSION", "v1"),
        )

        # 7) Métricas a Application Insights (dims Layer + ChunkHash)
        m_length.record(report.length_chars, {"layer": layer, "chunk_hash": report.chunk_hash})
        m_entropy.record(report.shannon_entropy, {"layer": layer, "chunk_hash": report.chunk_hash})
        m_cos.record(report.cosine_to_centroid, {"layer": layer, "chunk_hash": report.chunk_hash})
        span.set_attribute("layer", layer)
        span.set_attribute("alerta_t", alerta_t)
        span.set_attribute("quarantine", quarantine)

        # 8) Side-effects: quarantine a Blob + evento a Event Grid
        if quarantine:
            _QUARANTINE_TS.append(datetime.now(timezone.utc))
            _quarantine_blob(chunk.name, text, report)
            _EVENTGRID.send([{
                "eventType": "Autodoc.ChunkQuarantined",
                "subject": f"autodoc-chunks/{chunk.name}",
                "data": report.__dict__,
                "eventTime": datetime.now(timezone.utc).isoformat(),
                "dataVersion": "1",
            }])
            logger.warning("chunk quarantined", extra=report.__dict__)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Kill switch reactivo — Event Grid trigger (sustituye a EventBridge/SNS)
#    Misma POLÍTICA: alerta_t=1 + rolling 24h → freeze_retrieval_lambda
#    (en Azure: congela retrieval Function / deshabilita el plan de Pipelines).
# ──────────────────────────────────────────────────────────────────────────────
@app.route(route="killswitch", methods=["POST"])
@app.event_grid_trigger(arg_name="event")
def killswitch(event: func.EventGridEvent) -> None:
    """Reactivo (NO cron) sobre `autodoc-chunks` writes; igual que EventBridge.

    Si la rolling-window 24h supera el umbral → Logic Apps publica en
    una cola `freeze-retrieval` y el Function de retrieval se deshabilita.
    """
    if event.event_type != "Autodoc.ChunkQuarantined":
        return
    rate = _compute_quarantine_rate_24h()
    if rate > 0.10:  # umbral operativo (análogo al de ep.9)
        # En el repo del curso: Logic Apps + Service Bus; aquí el contrato.
        logger.error("killswitch_engaged", extra={"rate_24h": rate,
                                                  "action": "freeze_retrieval_function"})


# ──────────────────────────────────────────────────────────────────────────────
# 8. Helpers (movidos a autodoc_core en el repo; aquí, mínimos para demo)
# ──────────────────────────────────────────────────────────────────────────────
def _layer_from_blob_name(name: str) -> str:
    # autodoc-chunks/<layer>/<file>; ej. "finanzas/q3.txt" → "finanzas"
    parts = name.split("/")
    return parts[1] if len(parts) >= 3 else "interno"

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    p = np.fromiter((c / len(s) for c in Counter(s).values()), dtype=float)
    return float(-(p * np.log2(p)).sum())

def _valida_citacion(text: str, layer: str) -> bool:
    # Misma regla del ep.9 — citas "28 EUR/día" válidas, "128 EUR/día" no.
    import re
    matches = re.findall(r"\b\d+\s*EUR/día\b", text)
    return all(m.strip().startswith("28 ") for m in matches)

def _quarantine_blob(name: str, text: str, report: DriftReport) -> None:
    svc = BlobServiceClient.from_connection_string(os.environ["AZURE_BLOB_CONNECTION_STRING"])
    container = svc.get_container_client("autodoc-quarantine")
    target = f"{datetime.now(timezone.utc):%Y-%m-%d}/{report.chunk_hash}.txt"
    container.upload_blob(target, text + "\n\n# " + json.dumps(report.__dict__),
                          overwrite=True, content_type="text/plain")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Demo end-to-end — los 8 chunks del ep.9, ahora ejecutados en Azure
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Replica local del demo: 8 chunks; ahorro 72% por híbrida (2-3 selectivos vs 8).
    chunks = [
        "Coste medio de infraestructura: 28 EUR/día en Q3.",     # válido
        "Presupuesto aprobado: 28 EUR/día por nodo.",            # válido
        "El precio de la tortilla en Madrid ronda 1,2 EUR.",     # ruido k-NN
        "TODO: revisar cifras antes de publicar.",               # ruido regex
        "Estimación interna 28 EUR/día — pendiente validar.",    # válido
        "Coste outlier detectado: 128 EUR/día.",                 # FALSO → escalado
        "Ticket #TODO cerrar antes del cierre trimestral.",      # ruido regex
        "Anomalía histórica con z=4.2 en consumo eléctrico.",   # ruido MAD
    ]
    # En el curso: Function Core Tools `func start` + `az storage blob upload`.
    # Aquí imprimimos los DriftReport que la pipeline produciría:
    for i, t in enumerate(chunks, 1):
        alerta = 1 if i in (3, 4, 6, 7, 8) else 0  # sim. d_i vs τ
        valida = _valida_citacion(t, "finanzas")
        print(f"[{i}] alerta_t={alerta}  válida={valida}  quarantine={bool(alerta or not valida)}  :: {t[:60]}")