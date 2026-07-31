"""
Track 3: RAG Pipeline logic for image embedding and semantic search.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import pyarrow as pa
import lancedb
from PIL import Image
from transformers import AutoProcessor, AutoModel

from app.config import settings

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing RAG pipeline on device: {self.device}")
        
        # Load SigLIP model and processor
        self.model_name = settings.rag.model_name
        logger.info(f"Loading SigLIP model: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # Connect to LanceDB
        logger.info(f"Connecting to LanceDB at {settings.paths.lancedb_path}")
        settings.paths.lancedb_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(settings.paths.lancedb_path))
        
        # The SigLIP base patch16 embedding dimension is 768
        self.dim = self.model.config.text_config.hidden_size
        
        self.schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), self.dim)),
            pa.field("event_id", pa.string()),
            pa.field("object_id", pa.string()),
            pa.field("image_path", pa.string())
        ])
        
        self.table_name = settings.rag.table_name
        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=self.schema)
        else:
            self.table = self.db.open_table(self.table_name)
            
    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Compute normalized image embeddings."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        image_features = out.pooler_output if hasattr(out, 'pooler_output') else (out[0] if isinstance(out, tuple) else out)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        """Compute normalized text embeddings."""
        inputs = self.processor(text=text, padding="max_length", return_tensors="pt").to(self.device)
        out = self.model.get_text_features(**inputs)
        text_features = out.pooler_output if hasattr(out, 'pooler_output') else (out[0] if isinstance(out, tuple) else out)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].cpu().numpy().tolist()

    def ingest_event_crops(self, event_id: str) -> int:
        """Finds all entity crops for an event and ingests them into LanceDB."""
        crops_dir = settings.paths.dataset_dir / event_id / "entity_crops"
        if not crops_dir.exists():
            logger.warning(f"No crops directory found for event {event_id}")
            return 0
            
        jpg_files = list(crops_dir.glob("*.jpg"))
        if not jpg_files:
            return 0

        # Idempotent ingest: drop any previously-indexed rows for this event so
        # re-indexing replaces them instead of piling up duplicate vectors.
        try:
            self.table.delete(f"event_id = '{event_id}'")
        except Exception as e:
            logger.warning(f"Could not clear existing rows for {event_id}: {e}")

        records = []
        batch_size = 32
        
        for i in range(0, len(jpg_files), batch_size):
            batch_files = jpg_files[i:i+batch_size]
            images = []
            meta = []
            
            for f in batch_files:
                try:
                    img = Image.open(f).convert("RGB")
                    images.append(img)
                    
                    # Parse object_id: {event_id}_{object_id}_crop.jpg
                    fname = f.stem
                    prefix = f"{event_id}_"
                    suffix = "_crop"
                    
                    obj_id = fname
                    if obj_id.startswith(prefix):
                        obj_id = obj_id[len(prefix):]
                    if obj_id.endswith(suffix):
                        obj_id = obj_id[:-len(suffix)]
                        
                    meta.append({
                        "event_id": event_id,
                        "object_id": obj_id,
                        # Store relative to base dataset path for portability
                        "image_path": str(f.relative_to(settings.paths.base_dir).as_posix())
                    })
                except Exception as e:
                    logger.error(f"Failed to load image {f}: {e}")
                    
            if not images:
                continue
                
            embeddings = self.embed_images(images)
            for j, emb in enumerate(embeddings):
                record = meta[j]
                record["vector"] = emb
                records.append(record)
                
        if records:
            self.table.add(records)
            logger.info(f"Ingested {len(records)} crops for event {event_id} into LanceDB")
            
        return len(records)
        
    @torch.no_grad()
    def zero_shot_batch(self, images: list, labels: list[str]) -> list[str]:
        """Zero-shot classify each image against `labels` (SigLIP cosine similarity).

        Returns the best-matching label per image. Used by Track 4 to derive
        text attributes (colour, type) for entities, keeping the LLM text-only.
        """
        if not images:
            return []
        img = np.asarray(self.embed_images(images), dtype=np.float32)          # (N, dim)
        lab = np.asarray([self.embed_text(l) for l in labels], dtype=np.float32)  # (L, dim)
        best = (img @ lab.T).argmax(axis=1)
        return [labels[i] for i in best]

    def index_vehicles(self, source_id: str, items: list) -> int:
        """
        Index continuously-observed vehicles (hybrid all-vehicle corpus) into LanceDB.

        `items` is a list of (track_id, PIL.Image, image_path). Records share the
        existing schema with `event_id` holding the per-feed bucket (source_id) and
        `object_id` the track id. Cross-track duplicates of the same vehicle are
        handled at search time by the existing de-duplication.
        """
        if not items:
            return 0
        images = [im for _tid, im, _path in items]
        embeddings = self.embed_images(images)
        records = []
        for (tid, _im, path), emb in zip(items, embeddings):
            records.append({"vector": emb, "event_id": source_id,
                            "object_id": f"V_{int(tid):04d}", "image_path": path})
        self.table.add(records)
        logger.info(f"Indexed {len(records)} vehicles for {source_id} into LanceDB")
        return len(records)

    def search_crops(self, query: str, limit: int = 5,
                     dedup: bool = True, dup_sim: float = 0.95) -> list[dict]:
        """Semantic search over entity crops, with optional de-duplication.

        With dedup on, over-fetches then collapses (a) exact repeats of the same
        (event_id, object_id) — e.g. from repeated ingests — and (b) near-identical
        embeddings (cosine > dup_sim), which are the same physical entity fragmented
        into multiple track IDs. Results arrive sorted by ascending distance, so the
        best match of each duplicate group is the one kept. Returns up to `limit`
        distinct entities.
        """
        vector = self.embed_text(query)
        fetch = max(limit * 5, limit) if dedup else limit
        hits = self.table.search(vector).limit(fetch).to_list()
        if not dedup:
            return [self._clean_hit(h) for h in hits[:limit]]
        return self._dedup_hits(hits, limit, dup_sim)

    @staticmethod
    def _clean_hit(h: dict) -> dict:
        """Project a raw LanceDB hit to the API result shape (drop the vector)."""
        return {
            "event_id": h["event_id"],
            "object_id": h["object_id"],
            "image_path": h["image_path"],
            "distance": h.get("_distance", 0.0),
        }

    @staticmethod
    def _dedup_hits(hits: list[dict], limit: int, dup_sim: float) -> list[dict]:
        """Collapse duplicate hits, keeping the closest match per entity.

        Drops exact (event_id, object_id) repeats and near-identical embeddings
        (cosine > dup_sim). Assumes `hits` is ordered best-first and that stored
        vectors are L2-normalised, so a dot product is the cosine similarity.
        """
        kept: list[dict] = []
        kept_vecs: list[np.ndarray] = []
        seen_pairs: set[tuple[str, str]] = set()
        for h in hits:
            key = (h["event_id"], h["object_id"])
            if key in seen_pairs:
                continue
            vec = h.get("vector")
            v = np.asarray(vec, dtype=np.float32) if vec is not None else None
            if v is not None and any(float(v @ kv) > dup_sim for kv in kept_vecs):
                continue
            seen_pairs.add(key)
            kept.append(RAGPipeline._clean_hit(h))
            if v is not None:
                kept_vecs.append(v)
            if len(kept) >= limit:
                break
        return kept

# Singleton instance
_pipeline_instance = None

def get_rag_pipeline() -> RAGPipeline:
    """Lazy init to avoid loading weights on module import."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
