# autodoc_core/pipeline.py  (idéntico en ambos episodios)
from autodoc_core.ports import CentroidStore

class DefensiveOutlierPipeline:
    def __init__(self, embedder, centroid_store: CentroidStore,
                 drop_mode: str = "quarantine"):
        self.embedder = embedder
        self.store = centroid_store          # <- provider-agnostic
        self.drop_mode = drop_mode

    def run(self, chunks, vecs, layer, date):
        c_bar = self.store.load(layer, date)         # load indiferente
        # ... MAD z-score sobre vecs, regex `#TODO`, k-NN con d>c̄ ...
        kept, quarantined = self._filter(chunks, vecs, c_bar)
        self.store.put(layer, date, self._recompute_centroid(kept))
        return kept, quarantined