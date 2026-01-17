"""
Hugging Face Spaces entrypoint for Wisdom Ai (Gradio UI).
"""
import os
import torch
import gradio_ui as core


# Optional RAG warmup (non-fatal if embedder not available)
try:
    device_pref = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        core.load_embedder(device=device_pref)
        try:
            _ = core._rag_embedder.encode(["warmup"], convert_to_tensor=True, device=device_pref)
        except Exception:
            pass
    except Exception:
        pass
    core.build_or_load_rag_index(device=device_pref)
except Exception:
    pass


# Load model using environment overrides for deployments
base_model = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
adapter_dir = os.getenv("ADAPTER_ID", "./lora_mistral_bhagavad")

core.tokenizer, core.model = core.load_model(base_model, adapter_dir)

demo = core.create_ui(share=False, streaming=True)

if __name__ == "__main__":
    demo.launch()