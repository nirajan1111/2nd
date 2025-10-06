import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Tuple


class NeuralLegalParser(nn.Module):
    """
    T5-based Neural Parser for Legal Clauses
    Converts natural language legal text to FOPL (First-Order Predicate Logic)
    """
    
    def __init__(self, model_name: str = "t5-base", max_length: int = 512):
        super(NeuralLegalParser, self).__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Load T5 model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add custom tokens for FOPL (as regular tokens, not special tokens)
        new_tokens = [
            'forall', 'exists', '&', '|', '~', '->', '<->', '<=', '>=', '!=',
            '(', ')', ',', '[', ']',
            'Tenant', 'Landlord', 'Buyer', 'Seller', 'Employee', 
            'Contractor', 'Supplier', 'Client',
            'PayRent', 'Terminate', 'Maintain', 'Deliver',
            'Liable', 'Indemnify', 'Warranty'
        ]
        # Add tokens to vocabulary (not as special tokens!)
        num_added = self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"✅ Loaded {model_name} with {len(self.tokenizer)} tokens")
    
    def preprocess_input(self, clause_text: str, context: Dict = None) -> str:
        """
        Preprocess legal clause text for T5 input
        Format: "translate legal to logic: <clause> context: <entities>"
        """
        input_text = f"translate legal to logic: {clause_text}"
        
        if context:
            context_str = " ".join([f"{k}={v}" for k, v in context.items()])
            input_text += f" context: {context_str}"
        
        return input_text
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor = None) -> Dict:
        """Forward pass through T5 model"""
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits
        }
    
    def parse(self, clause_text: str, context: Dict = None, 
              num_beams: int = 4, temperature: float = 0.7) -> str:
        """
        Parse legal clause to FOPL representation
        
        Args:
            clause_text: Natural language legal clause
            context: Dictionary of entity mappings
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            FOPL representation string
        """
        self.model.eval()
        
        # Preprocess input
        input_text = self.preprocess_input(clause_text, context)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate FOPL
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode (keep special tokens to preserve FOPL structure)
        fopl_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Clean up extra tokens (</s>, <pad>)
        fopl_output = fopl_output.replace('</s>', '').replace('<pad>', '').strip()
        
        return fopl_output
    
    def batch_parse(self, clause_texts: List[str], contexts: List[Dict] = None,
                    batch_size: int = 8) -> List[str]:
        """Parse multiple clauses in batches"""
        
        if contexts is None:
            contexts = [None] * len(clause_texts)
        
        results = []
        
        for i in range(0, len(clause_texts), batch_size):
            batch_texts = clause_texts[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            
            # Preprocess batch
            input_texts = [
                self.preprocess_input(text, ctx) 
                for text, ctx in zip(batch_texts, batch_contexts)
            ]
            
            # Tokenize
            inputs = self.tokenizer(
                input_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode (keep special tokens to preserve FOPL structure)
            batch_results = [
                self.tokenizer.decode(out, skip_special_tokens=False).replace('</s>', '').replace('<pad>', '').strip()
                for out in outputs
            ]
            results.extend(batch_results)
        
        return results
    
    def save_pretrained(self, save_path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✅ Model saved to {save_path}")
    
    def load_pretrained(self, load_path: str):
        """Load model and tokenizer"""
        self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        self.tokenizer = T5Tokenizer.from_pretrained(load_path)
        print(f"✅ Model loaded from {load_path}")


class FOPLValidator:
    """Validates generated FOPL syntax"""
    
    @staticmethod
    def validate(fopl_string: str) -> Tuple[bool, str]:
        """
        Validate FOPL syntax
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check balanced parentheses
            if fopl_string.count('(') != fopl_string.count(')'):
                return False, "Unbalanced parentheses"
            
            # Check for required quantifiers
            if not any(q in fopl_string for q in ['forall', 'exists']):
                return False, "Missing quantifier (forall/exists)"
            
            # Check for predicates
            if not any(c.isupper() for c in fopl_string):
                return False, "No predicates found"
            
            return True, "Valid FOPL"
            
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def extract_predicates(fopl_string: str) -> List[str]:
        """Extract all predicates from FOPL string"""
        import re
        
        # Pattern: CapitalizedWord followed by parentheses
        pattern = r'[A-Z][a-zA-Z]*\([^)]*\)'
        predicates = re.findall(pattern, fopl_string)
        
        return list(set(predicates))


if __name__ == "__main__":
    # Test the neural parser
    parser = NeuralLegalParser(model_name="t5-small")
    
    # Test clause
    clause = "The tenant must pay rent by the 5th of each month."
    context = {"Tenant": "PartyA", "Landlord": "PartyB"}
    
    print(f"Input: {clause}")
    print(f"Context: {context}")
    print(f"Preprocessed: {parser.preprocess_input(clause, context)}")
    
    # Test FOPL validator
    validator = FOPLValidator()
    test_fopl = "forall x (Tenant(x) -> PayRent(x, due_date <= 5))"
    is_valid, msg = validator.validate(test_fopl)
    print(f"\nFOPL: {test_fopl}")
    print(f"Valid: {is_valid}, Message: {msg}")
    print(f"Predicates: {validator.extract_predicates(test_fopl)}")