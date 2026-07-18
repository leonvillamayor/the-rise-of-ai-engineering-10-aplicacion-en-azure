# function_app.py  (plan Consumption, Python 3.12)
import azure.functions as func
from autodoc_core import DefensiveOutlierPipeline, DriftReport
from autodoc_azureci import AzureAIFoundryEmbedder, AzureBlobCentroidStore
from autodoc_core.metrics import quarantine_rate_24h

app = func.FunctionApp()

@app.blob_trigger(arg_name="blob", path="autodoc-chunks/{name}",
                  connection="CHUNKS_STORAGE")
@app.function_name(name="indexer")
def indexer(blob: func.InputStream) -> None:
    raw = blob.read().decode("utf-8")
    chunks = splitter.split(raw)                       # misma lógica que ep.9

    # (1) Embeddings via AI Foundry — mismo contrato embed(text)->Vector
    embedder = AzureAIFoundryEmbedder(model="text-embedding-3-large-1024")
    vecs = [embedder.embed(c.text) for c in chunks]

    # (2)-(3) Pipeline defensiva — código idéntico al ep.9
    pipe = DefensiveOutlierPipeline(
        embedder=embedder,
        centroid_store=AzureBlobCentroidStore(         # (3) adapter, no policy
            account_url=os.environ["CENTROIDS_URL"],
            container="centroids",
            sas=os.environ["CENTROIDS_SAS"],
        ),
        drop_mode="quarantine",                        # <- intacto
    )
    kept, quarantined = pipe.run(chunks, vecs, layer="L1", date=today)

    # Drift d_i = 1 - cos(c_i, c̄) — INDEXER, nunca query-time
    c_bar = pipe.centroid()
    report = DriftReport(
        chunks=[c.id for c in kept],
        d_i=[1 - cosine(c.v, c_bar) for c in kept],    # fórmula portable
        quarantine_rate_24h=quarantine_rate_24h(),
    )
    report.publish_to_app_insights()