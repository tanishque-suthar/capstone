"""
Track 3: RAG Pipeline logic for image embedding and semantic search.
"""

import logging
from pathlib import Path

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
        
    def search_crops(self, query: str, limit: int = 5) -> list[dict]:
        """Search LanceDB table for matching image crops."""
        vector = self.embed_text(query)
        results = self.table.search(vector).limit(limit).to_list()
        
        # Clean results (drop vector)
        clean_results = []
        for r in results:
            clean_results.append({
                "event_id": r["event_id"],
                "object_id": r["object_id"],
                "image_path": r["image_path"],
                "distance": r.get("_distance", 0.0)
            })
            
        return clean_results

# Singleton instance
_pipeline_instance = None

def get_rag_pipeline() -> RAGPipeline:
    """Lazy init to avoid loading weights on module import."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
