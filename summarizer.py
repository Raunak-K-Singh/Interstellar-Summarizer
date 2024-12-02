import torch
from typing import List, Dict, Optional
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    pipeline, 
    MarianMTModel, 
    MarianTokenizer
)

class AdvancedCrossLingualSummarizer:
    def __init__(self, 
                 summary_model='google/mt5-large', 
                 translation_model='Helsinki-NLP/opus-mt'):
        """
        Initialize advanced cross-lingual summarization system.
        """
        # Comprehensive language mapping
        self.language_codes = {
            'en': {'code': 'en_XX', 'name': 'English'},
            'es': {'code': 'es_XX', 'name': 'Spanish'},
            'fr': {'code': 'fr_XX', 'name': 'French'},
            'de': {'code': 'de_DE', 'name': 'German'},
            'it': {'code': 'it_IT', 'name': 'Italian'},
            'ru': {'code': 'ru_RU', 'name': 'Russian'},
            'zh': {'code': 'zh_CN', 'name': 'Chinese'},
            'ar': {'code': 'ar_AR', 'name': 'Arabic'},
            'hi': {'code': 'hi_IN', 'name': 'Hindi'},
            'sa': {'code': 'sa_IN', 'name': 'Sanskrit'}
        }
        
        # Load summarization model
        self.summary_tokenizer = AutoTokenizer.from_pretrained(summary_model)
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model)
        
        # Translation pipelines for additional language support
        self.translation_pipelines = {}
        self.summarization_strategies = ['extractive', 'abstractive', 'hybrid']
    
    def _load_translation_pipeline(self, source_lang: str, target_lang: str):
        translation_key = f"{source_lang}_{target_lang}"
        if translation_key not in self.translation_pipelines:
            try:
                model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
                translation_model = MarianMTModel.from_pretrained(model_name)
                translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
                self.translation_pipelines[translation_key] = pipeline(
                    'translation', 
                    model=translation_model, 
                    tokenizer=translation_tokenizer
                )
            except Exception as e:
                print(f"Failed to load translation pipeline: {e}")
                return None
        return self.translation_pipelines[translation_key]
    
    def summarize(self, text: str, source_lang: str = 'en', target_lang: str = 'en',
                  strategy: str = 'abstractive', max_length: int = 150, 
                  min_length: int = 50, num_return_sequences: int = 3) -> List[str]:
        """
        Advanced cross-lingual summarization with multiple strategies.
        """
        if source_lang not in self.language_codes or target_lang not in self.language_codes:
            raise ValueError("Unsupported language")
        if strategy not in self.summarization_strategies:
            raise ValueError("Invalid strategy")
        
        if source_lang != target_lang:
            translation_pipeline = self._load_translation_pipeline(source_lang, target_lang)
            if translation_pipeline:
                text = translation_pipeline(text, max_length=1024)[0]['translation_text']
        
        inputs = self.summary_tokenizer(
            text, max_length=1024, truncation=True, return_tensors="pt"
        )
        
        summaries = []
        for _ in range(num_return_sequences):
            summary_ids = self.summary_model.generate(
                inputs['input_ids'], 
                num_beams=4, 
                max_length=max_length, 
                min_length=min_length,
                do_sample=True, 
                temperature=0.7, 
                top_k=50, 
                top_p=0.95
            )
            summary = self.summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        return summaries
    
    def list_supported_languages(self) -> List[Dict]:
        return [{'code': code, 'name': details['name']} for code, details in self.language_codes.items()]
