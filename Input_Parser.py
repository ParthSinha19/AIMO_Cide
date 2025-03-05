import os
import re
import json
import zipfile
import tempfile
import traceback

import PyPDF2
import torch
import numpy as np
from typing import List, Dict, Any
import fitz  # PyMuPDF for better PDF image extraction
from PIL import Image
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    ViTImageProcessor,
    ViTForImageClassification,
    ViTFeatureExtractor
)
class MathProblemParser:
    def __init__(self, models_config: Dict[str, str] = None):
        
        self.default_models = {
            "nlp_model": "facebook/bart-base",  
            "text_extraction": "microsoft/layoutlm-base-uncased"
        }
        
        
        self.models_config = models_config or self.default_models
        
        
        if pipeline is None or AutoTokenizer is None:
            print("Warning: Transformers pipeline not available. Using fallback methods.")
            self.nlp_pipeline = None
            self.text_extractor = None
            return
        
        try:
            
            print("Loading NLP model...")
            self.nlp_pipeline = pipeline(
                "text2text-generation",
                model=self.models_config["nlp_model"],
                device=-1,  
                model_kwargs={"low_cpu_mem_usage": True}
            )
            print("NLP model loaded successfully")
            
            
            try:
                print("Loading text extraction model...")
                tokenizer = AutoTokenizer.from_pretrained(self.models_config["text_extraction"])
                
                
                model = AutoModelForTokenClassification.from_pretrained(
                    self.models_config["text_extraction"],
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                ).to('cpu')  
                
                
                self.text_extractor = pipeline(
                    "token-classification",
                    model=model.to('cpu'), 
                    tokenizer=tokenizer,
                    device=-1,
                    aggregation_strategy="simple"
                )
                print("Text extraction model loaded successfully")
            
            except Exception as model_init_error:
                print(f"Error initializing text extraction model: {model_init_error}")
                print("Falling back to default text processing methods.")
                self.text_extractor = None
        
        except Exception as e:
            print(f"Critical error in model initialization: {e}")
            self.nlp_pipeline = None
            self.text_extractor = None
    def extract_pdf_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract structured text from PDF
        """
        try:
            
            pdf_file = open(pdf_path, 'rb')
            reader = PyPDF2.PdfReader(pdf_file)
            
            document_info = {
                "metadata": {},
                "problems": []
            }
            
            
            document_info["metadata"] = {
                "title": reader.metadata.get('/Title', 'Unknown'),
                "filename": os.path.basename(pdf_path),
                "total_pages": len(reader.pages)
            }
            
            
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n\n"
            
            
            pdf_file.close()
            
            
            problem_matches = re.findall(r'Problem (\d+)\.\s*(.*?)(?=Problem \d|\Z)', full_text, re.DOTALL)
            
            for problem_num, problem_text in problem_matches:
                problem_info = {
                    "number": int(problem_num),
                    "text": problem_text.strip(),
                    "entities": self.extract_entities(problem_text),
                    "constraints": self.map_constraints(problem_text)
                }
                document_info["problems"].append(problem_info)
            
            return document_info
        
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            
            if 'pdf_file' in locals():
                pdf_file.close()
            return {
                "metadata": {
                    "filename": os.path.basename(pdf_path),
                    "error": str(e)
                }, 
                "problems": []
            }
    
    def extract_entities(self, problem_text: str) -> List[Dict[str, Any]]:
       
        entities = []
        
        
        patterns = {
            "integer": r'\b(\d+)\b',
            "variable": r'\b([a-zA-Z])\b',
            "gcd": r'gcd\(',
            "triangle": r'triangle\s+([A-Z]{3})',
            "sequence": r'sequence\s+([a-z]\d*)'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, problem_text)
            for match in matches:
                entities.append({
                    "type": entity_type,
                    "value": match
                })
        
        return entities
    
    def map_constraints(self, problem_text: str) -> List[Dict[str, Any]]:
        
        constraints = []
        
        
        constraint_rules = [
            {
                "pattern": r"for all (.*?),\s*(.*?)$",
                "type": "universal_quantifier"
            },
            {
                "pattern": r"there exists (.*?),\s*(.*?)$",
                "type": "existential_quantifier"
            }
        ]
        
        for rule in constraint_rules:
            matches = re.findall(rule["pattern"], problem_text, re.IGNORECASE)
            for match in matches:
                constraints.append({
                    "type": rule["type"],
                    "variables": match[0],
                    "condition": match[1]
                })
        
        return constraints
    
    def process_zip_file(self, zip_path: str) -> List[Dict[str, Any]]:
        parsed_documents = []
        
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            except Exception as zip_error:
                print(f"Error extracting ZIP file: {zip_error}")
                return []
            
            
            pdf_files = [f for f in os.listdir(temp_dir) if f.lower().endswith('.pdf')]
            
            print(f"Found {len(pdf_files)} PDF files to process")
            
            for filename in pdf_files:
                full_path = os.path.join(temp_dir, filename)
                try:
                    document_info = self.extract_pdf_text(full_path)
                    parsed_documents.append(document_info)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return parsed_documents
    
    def export_parsed_documents(self, parsed_documents: List[Dict[str, Any]], output_dir: str):
        
        
        os.makedirs(output_dir, exist_ok=True)
        
        for doc in parsed_documents:
            
            filename = doc['metadata'].get('filename', f"document_{hash(str(doc))}.json")
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_parsed.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(parsed_documents)} documents to {output_dir}")
def get_entity_labels(self) -> List[str]:
    
    return [
        "O",  
        "B-INTEGER", "I-INTEGER",
        "B-VARIABLE", "I-VARIABLE",
        "B-OPERATION", "I-OPERATION",
        "B-CONSTRAINT", "I-CONSTRAINT"
    ]

def train_model(self, training_data: List[Dict[str, Any]], epochs: int = 3):
    if not self.text_extractor:
        print("Text extraction model not initialized")
        return

    try:
        from transformers import TrainingArguments, Trainer
        import torch.nn as nn
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir="./model_checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir="./logs",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.prepare_dataset(training_data),
        )
        
        # Train the model
        trainer.train()
        
        # Update pipeline with trained model
        self.text_extractor.model = self.model
        
        print("Model training completed successfully")
    
    except Exception as e:
        print(f"Error during model training: {e}")
    def extract_pdf_text(self, pdf_path: str) -> Dict[str, Any]:
        try:
            
            pdf_file = open(pdf_path, 'rb')
            reader = PyPDF2.PdfReader(pdf_file)
            
            document_info = {
                "metadata": {},
                "problems": []
            }
            
            
            document_info["metadata"] = {
                "title": reader.metadata.get('/Title', 'Unknown'),
                "filename": os.path.basename(pdf_path),
                "total_pages": len(reader.pages)
            }
            
            
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n\n"
            
            
            pdf_file.close()
            
            
            problem_matches = re.findall(r'Problem (\d+)\.\s*(.*?)(?=Problem \d|\Z)', full_text, re.DOTALL)
            
            for problem_num, problem_text in problem_matches:
                problem_info = {
                    "number": int(problem_num),
                    "text": problem_text.strip(),
                    "entities": self.extract_entities(problem_text),
                    "constraints": self.map_constraints(problem_text)
                }
                document_info["problems"].append(problem_info)
            
            return document_info
        
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            
            if 'pdf_file' in locals():
                pdf_file.close()
            return {
                "metadata": {
                    "filename": os.path.basename(pdf_path),
                    "error": str(e)
                }, 
                "problems": []
            }
    
    def extract_entities(self, problem_text: str) -> List[Dict[str, Any]]:
        
        entities = []
        
        
        patterns = {
            "integer": r'\b(\d+)\b',
            "variable": r'\b([a-zA-Z])\b',
            "gcd": r'gcd\(',
            "triangle": r'triangle\s+([A-Z]{3})',
            "sequence": r'sequence\s+([a-z]\d*)'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, problem_text)
            for match in matches:
                entities.append({
                    "type": entity_type,
                    "value": match
                })
        
        return entities
    
    def map_constraints(self, problem_text: str) -> List[Dict[str, Any]]:
        constraints = []
        
        
        constraint_rules = [
            {
                "pattern": r"for all (.*?),\s*(.*?)$",
                "type": "universal_quantifier"
            },
            {
                "pattern": r"there exists (.*?),\s*(.*?)$",
                "type": "existential_quantifier"
            }
        ]
        
        for rule in constraint_rules:
            matches = re.findall(rule["pattern"], problem_text, re.IGNORECASE)
            for match in matches:
                constraints.append({
                    "type": rule["type"],
                    "variables": match[0],
                    "condition": match[1]
                })
        
        return constraints
    
    def process_zip_file(self, zip_path: str) -> List[Dict[str, Any]]:
        parsed_documents = []
        
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            except Exception as zip_error:
                print(f"Error extracting ZIP file: {zip_error}")
                return []
            
            
            pdf_files = [f for f in os.listdir(temp_dir) if f.lower().endswith('.pdf')]
            
            print(f"Found {len(pdf_files)} PDF files to process")
            
            for filename in pdf_files:
                full_path = os.path.join(temp_dir, filename)
                try:
                    document_info = self.extract_pdf_text(full_path)
                    parsed_documents.append(document_info)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return parsed_documents
    
    def export_parsed_documents(self, parsed_documents: List[Dict[str, Any]], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        for doc in parsed_documents:
            
            filename = doc['metadata'].get('filename', f"document_{hash(str(doc))}.json")
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_parsed.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(parsed_documents)} documents to {output_dir}")
def main():
    try:
        
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        
        try:
            parser = MathProblemParser()
            if parser.nlp_pipeline is None:
                print("Warning: NLP pipeline initialization failed. Check model availability and dependencies.")
        except Exception as model_error:
            print(f"Error initializing MathProblemParser: {model_error}")
            return
    
        
        zip_file_path = '/content/Problems_Cide.zip'
        output_directory = '/content/OutputFolder'
        
        
        os.makedirs(output_directory, exist_ok=True)
        
        
        if not os.path.exists(zip_file_path):
            print(f"Error: ZIP file not found at {zip_file_path}")
            return
        
        
        parsed_documents = parser.process_zip_file(zip_file_path)
        
        
        if not parsed_documents:
            print("No documents were parsed. Check the ZIP file contents and file paths.")
            return
        
        
        parser.export_parsed_documents(parsed_documents, output_directory)
        
        
        for doc in parsed_documents:
            print(f"Document: {doc['metadata'].get('filename', 'Unknown')}")
            for problem in doc.get('problems', []):
                print(f"Problem {problem.get('number', 'N/A')}")
                print(f"Entities: {len(problem.get('entities', []))}")
                print(f"Constraints: {len(problem.get('constraints', []))}")
                print("-" * 50)
    
    except Exception as e:
        print(f"Error processing ZIP file: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
