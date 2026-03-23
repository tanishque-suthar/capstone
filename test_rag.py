import sys
from app.pipeline.rag import get_rag_pipeline

def test_rag():
    print("Testing RAG Pipeline Initialization...")
    pipeline = get_rag_pipeline()
    
    print(f"Device: {pipeline.device}")
    print(f"Model: {pipeline.model_name}")
    print(f"Database Table: {pipeline.table_name}")
    
    print("\nTesting Text Embedding...")
    query = "a white car"
    emb = pipeline.embed_text(query)
    print(f"Embedded '{query}', vector length: {len(emb)}")
    
    print("\nInitialization test success.")

if __name__ == "__main__":
    try:
        test_rag()
    except Exception as e:
        print(f"Error during RAG test: {e}")
        sys.exit(1)
