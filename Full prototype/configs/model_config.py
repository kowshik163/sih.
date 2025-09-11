"""
FRA AI System - Model Configuration
Centralized model definitions and configurations for all AI components
"""

# Primary LLM Models (Apache/Open Licensed)
PRIMARY_LLMS = {
    "mistral_7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.1",
        "license": "Apache 2.0",
        "size": "7B",
        "context_length": 4096,
        "recommended_use": "Primary inference, fine-tuning",
        "memory_requirement": "14GB",
        "tasks": ["ner", "qa", "spatial_queries", "policy_analysis"]
    },
    "llama2_7b": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "license": "Custom (Research/Commercial)",
        "size": "7B", 
        "context_length": 4096,
        "recommended_use": "Backup model, knowledge distillation teacher",
        "memory_requirement": "14GB",
        "tasks": ["qa", "document_analysis", "policy_reasoning"]
    },
    "falcon_7b": {
        "name": "tiiuae/falcon-7b-instruct",
        "license": "Apache 2.0",
        "size": "7B",
        "context_length": 2048,
        "recommended_use": "Lightweight deployment, edge inference",
        "memory_requirement": "14GB",
        "tasks": ["basic_qa", "text_classification"]
    }
}

# OCR and Document Processing Models
OCR_MODELS = {
    "layoutlmv3_base": {
        "name": "microsoft/layoutlmv3-base",
        "task": "document_understanding",
        "input_types": ["images", "text"],
        "max_image_size": [224, 224],
        "languages": ["multilingual"],
        "use_case": "FRA document structure recognition"
    },
    "layoutlmv3_large": {
        "name": "microsoft/layoutlmv3-large", 
        "task": "document_understanding",
        "input_types": ["images", "text"],
        "max_image_size": [224, 224],
        "languages": ["multilingual"],
        "use_case": "High-accuracy document processing"
    },
    "trocr_base": {
        "name": "microsoft/trocr-base-stage1",
        "task": "optical_character_recognition",
        "input_types": ["images"],
        "languages": ["english"],
        "use_case": "Text recognition in documents"
    },
    "trocr_large": {
        "name": "microsoft/trocr-large-stage1",
        "task": "optical_character_recognition", 
        "input_types": ["images"],
        "languages": ["english"],
        "use_case": "High-accuracy text recognition"
    }
}

# Named Entity Recognition Models
NER_MODELS = {
    "indic_bert": {
        "name": "ai4bharat/indic-bert",
        "task": "named_entity_recognition",
        "languages": ["hindi", "bengali", "telugu", "marathi", "gujarati", "tamil", "kannada", "malayalam", "oriya"],
        "entities": ["PERSON", "LOCATION", "ORGANIZATION", "DATE"],
        "use_case": "Indian language NER for FRA documents"
    },
    "legal_ner": {
        "name": "opennyaiorg/en_legal_ner_trf",
        "task": "legal_entity_recognition",
        "languages": ["english"],
        "entities": ["COURT", "PETITIONER", "RESPONDENT", "JUDGE", "DATE", "PROVISION"],
        "use_case": "Legal document entity extraction"
    },
    "multilingual_bert": {
        "name": "bert-base-multilingual-cased",
        "task": "token_classification",
        "languages": ["multilingual"],
        "use_case": "General purpose multilingual NER"
    }
}

# Computer Vision Models for Satellite Analysis
SATELLITE_MODELS = {
    "deeplabv3_resnet50": {
        "name": "facebook/deeplabv3-resnet50-ade",
        "task": "semantic_segmentation",
        "input_size": [512, 512],
        "num_classes": 150,
        "use_case": "Land use classification from satellite imagery"
    },
    "deeplabv3_satellite": {
        "name": "qualcomm/DeepLabV3-ResNet50", 
        "task": "land_cover_segmentation",
        "input_size": [256, 256],
        "classes": ["forest", "agriculture", "water", "built_up", "barren"],
        "use_case": "Specialized satellite land cover mapping"
    },
    "segformer": {
        "name": "nvidia/segformer-b0-finetuned-ade-512-512",
        "task": "semantic_segmentation",
        "input_size": [512, 512], 
        "use_case": "Efficient segmentation for real-time processing"
    },
    "sam_model": {
        "name": "facebook/sam-vit-huge",
        "task": "image_segmentation",
        "input_size": [1024, 1024],
        "use_case": "High-quality asset boundary detection"
    },
    "prithvi_foundation": {
        "name": "ibm-nasa-geospatial/Prithvi-100M",
        "task": "geospatial_foundation",
        "input_size": [224, 224],
        "bands": ["multi_spectral"],
        "use_case": "Foundation model for satellite imagery analysis"
    }
}

# Object Detection Models
DETECTION_MODELS = {
    "detr_resnet50": {
        "name": "facebook/detr-resnet-50",
        "task": "object_detection",
        "input_size": [800, 1333],
        "num_classes": 91,
        "use_case": "Asset detection in satellite images"
    },
    "detr_panoptic": {
        "name": "facebook/detr-resnet-50-panoptic",
        "task": "panoptic_segmentation", 
        "input_size": [800, 1333],
        "use_case": "Comprehensive scene understanding"
    }
}

# Vision-Language Models
VISION_LANGUAGE_MODELS = {
    "clip_base": {
        "name": "openai/clip-vit-base-patch32",
        "task": "vision_language",
        "image_size": [224, 224],
        "text_length": 77,
        "use_case": "Image-text alignment and search"
    },
    "clip_large": {
        "name": "openai/clip-vit-large-patch14",
        "task": "vision_language",
        "image_size": [224, 224],
        "text_length": 77,
        "use_case": "High-quality multimodal understanding"
    }
}

# Language Models for Translation and Multilingual Processing
TRANSLATION_MODELS = {
    "legal_translation": {
        "name": "law-ai/InLegalTrans-En2Indic-1B",
        "task": "translation",
        "source_lang": "english",
        "target_langs": ["hindi", "bengali", "gujarati", "marathi", "telugu", "tamil", "kannada", "malayalam", "oriya"],
        "domain": "legal",
        "use_case": "Legal document translation for FRA"
    }
}

# Embedding Models for RAG and Similarity Search
EMBEDDING_MODELS = {
    "sentence_transformer": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "task": "text_embedding",
        "dimension": 384,
        "max_seq_length": 256,
        "use_case": "Document similarity and retrieval"
    },
    "legal_embeddings": {
        "name": "law-ai/InLegalBERT",
        "task": "legal_text_embedding",
        "dimension": 768,
        "domain": "legal",
        "use_case": "Legal document similarity and search"
    }
}

# Model Configuration Templates
MODEL_CONFIGS = {
    "inference": {
        "batch_size": 1,
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": 0
    },
    "training": {
        "batch_size": 4,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 4
    },
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
}

# Hardware Requirements
HARDWARE_REQUIREMENTS = {
    "minimum": {
        "gpu_memory": "8GB",
        "system_ram": "16GB", 
        "storage": "100GB",
        "gpu": "RTX 3070 / V100"
    },
    "recommended": {
        "gpu_memory": "24GB",
        "system_ram": "32GB",
        "storage": "500GB",
        "gpu": "RTX 4090 / A100"
    },
    "production": {
        "gpu_memory": "40GB+",
        "system_ram": "64GB+", 
        "storage": "1TB+",
        "gpu": "A100 / H100"
    }
}

# Task-Model Mapping
TASK_MODEL_MAPPING = {
    "document_ocr": ["layoutlmv3_base", "trocr_base"],
    "text_extraction": ["layoutlmv3_large", "trocr_large"],
    "named_entity_recognition": ["indic_bert", "legal_ner"],
    "satellite_segmentation": ["deeplabv3_satellite", "segformer"],
    "asset_detection": ["detr_resnet50", "sam_model"],
    "land_cover_classification": ["prithvi_foundation", "deeplabv3_resnet50"],
    "natural_language_queries": ["mistral_7b", "llama2_7b"],
    "policy_analysis": ["mistral_7b"],
    "document_qa": ["mistral_7b", "llama2_7b"],
    "multilingual_translation": ["legal_translation"],
    "document_similarity": ["sentence_transformer", "legal_embeddings"],
    "multimodal_understanding": ["clip_base", "clip_large"]
}
