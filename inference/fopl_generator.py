"""
FOPL Inference Module
Generates FOPL predicates from legal clauses using trained T5 model
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

logger = logging.getLogger(__name__)


class FOPLGenerator:
    """
    FOPL Predicate Generator using T5 model
    Converts legal clauses to First-Order Predicate Logic
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize FOPL generator
        
        Args:
            model_path: Path to trained T5 model (default: use config)
            device: Device to run on ('cuda', 'cpu', or 'mps')
        """
        # Load config
        cfg = get_config()
        
        # Set model path
        if model_path is None:
            model_path = str(cfg.paths.t5_fopl_checkpoint / "final")
            logger.info(f"Using default model path: {model_path}")
        
        self.model_path = model_path
        
        # Set device
        if device is None:
            device = cfg.get_device()
        self.device = torch.device(device)
        
        logger.info(f"Loading FOPL model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load trained model: {e}")
            logger.info("Falling back to base model")
            base_model = cfg.t5_fopl.model_name
            self.tokenizer = T5Tokenizer.from_pretrained(base_model)
            self.model = T5ForConditionalGeneration.from_pretrained(base_model)
            self.model.to(self.device)
            self.model.eval()
    
    def preprocess_input(self, clause_text: str, context: Optional[Dict[str, str]] = None) -> str:
        """
        Preprocess input for T5 model
        
        Args:
            clause_text: Legal clause text
            context: Entity context dictionary
        
        Returns:
            Formatted input string
        """
        input_text = f"translate to english logic: {clause_text}"
        
        if context:
            context_str = " ".join([f"{k}={v}" for k, v in context.items()])
            input_text += f" context: {context_str}"
        
        return input_text
    
    def generate(
        self,
        clause_text: str,
        context: Optional[Dict[str, str]] = None,
        num_beams: int = 4,
        max_length: int = 128,
        return_scores: bool = False
    ) -> Union[str, Dict]:
        """
        Generate FOPL predicate from legal clause
        
        Args:
            clause_text: Legal clause text
            context: Entity context
            num_beams: Number of beams for beam search
            max_length: Maximum output length
            return_scores: Whether to return confidence scores
        
        Returns:
            FOPL predicate string or dict with scores
        """
        # Preprocess
        input_text = self.preprocess_input(clause_text, context)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=True if num_beams > 1 else False,
                return_dict_in_generate=return_scores,
                output_scores=return_scores
            )
        
        if return_scores:
            # Extract sequences and scores
            sequences = outputs.sequences
            fopl_text = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
            
            # Calculate confidence (average score)
            scores = outputs.sequences_scores if hasattr(outputs, 'sequences_scores') else None
            confidence = float(scores[0]) if scores is not None else 0.0
            
            return {
                'fopl': fopl_text,
                'confidence': confidence,
                'input': clause_text
            }
        else:
            # Just return text
            fopl_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return fopl_text
    
    def batch_generate(
        self,
        clause_texts: List[str],
        contexts: Optional[List[Dict[str, str]]] = None,
        num_beams: int = 4,
        max_length: int = 128,
        batch_size: int = 8
    ) -> List[str]:
        """
        Generate FOPL predicates for multiple clauses
        
        Args:
            clause_texts: List of legal clause texts
            contexts: List of entity contexts (optional)
            num_beams: Number of beams for beam search
            max_length: Maximum output length
            batch_size: Batch size for processing
        
        Returns:
            List of FOPL predicate strings
        """
        if contexts is None:
            contexts = [None] * len(clause_texts)
        
        results = []
        
        # Process in batches
        for i in range(0, len(clause_texts), batch_size):
            batch_clauses = clause_texts[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            # Preprocess batch
            input_texts = [
                self.preprocess_input(clause, ctx)
                for clause, ctx in zip(batch_clauses, batch_contexts)
            ]
            
            # Tokenize batch
            inputs = self.tokenizer(
                input_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False,
                    early_stopping=True if num_beams > 1 else False
                )
            
            # Decode
            batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(batch_results)
        
        return results
    
    def validate_fopl(self, fopl_text: str) -> Dict:
        """
        Basic validation of FOPL syntax
        
        Args:
            fopl_text: FOPL predicate string
        
        Returns:
            Dict with validation results
        """
        issues = []
        
        # Check for basic structure
        if not fopl_text.strip():
            issues.append("Empty FOPL")
        
        # Check for quantifiers
        if 'forall' not in fopl_text and 'exists' not in fopl_text:
            issues.append("No quantifier found")
        
        # Check for balanced parentheses
        if fopl_text.count('(') != fopl_text.count(')'):
            issues.append("Unbalanced parentheses")
        
        # Check for predicates (capital letter followed by parentheses)
        import re
        predicates = re.findall(r'[A-Z][a-zA-Z]*\(', fopl_text)
        if not predicates:
            issues.append("No predicates found")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'predicates_found': len(predicates) if predicates else 0
        }
    
    def generate_with_validation(
        self,
        clause_text: str,
        context: Optional[Dict[str, str]] = None,
        num_beams: int = 4,
        max_length: int = 128
    ) -> Dict:
        """
        Generate FOPL with validation
        
        Args:
            clause_text: Legal clause text
            context: Entity context
            num_beams: Number of beams
            max_length: Maximum output length
        
        Returns:
            Dict with FOPL and validation results
        """
        # Generate
        result = self.generate(
            clause_text,
            context,
            num_beams=num_beams,
            max_length=max_length,
            return_scores=True
        )
        
        # Validate
        validation = self.validate_fopl(result['fopl'])
        
        return {
            **result,
            'validation': validation
        }


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def generate_fopl(
    clause_text: str,
    context: Optional[Dict[str, str]] = None,
    model_path: Optional[str] = None
) -> str:
    """
    Convenience function to generate FOPL from clause
    
    Args:
        clause_text: Legal clause text
        context: Entity context
        model_path: Path to trained model (optional)
    
    Returns:
        FOPL predicate string
    """
    generator = FOPLGenerator(model_path=model_path)
    return generator.generate(clause_text, context)


def batch_generate_fopl(
    clause_texts: List[str],
    contexts: Optional[List[Dict[str, str]]] = None,
    model_path: Optional[str] = None
) -> List[str]:
    """
    Convenience function for batch generation
    
    Args:
        clause_texts: List of legal clause texts
        contexts: List of entity contexts
        model_path: Path to trained model (optional)
    
    Returns:
        List of FOPL predicate strings
    """
    generator = FOPLGenerator(model_path=model_path)
    return generator.batch_generate(clause_texts, contexts)


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    # Test FOPL generator
    print("\n" + "="*60)
    print("TESTING FOPL GENERATOR")
    print("="*60 + "\n")
    
    # Initialize generator
    generator = FOPLGenerator()
    
    # Test cases
    test_cases = [
        {
            'clause': "The tenant must pay rent by the 5th of each month.",
            'context': {"Tenant": "PartyA"}
        },
        {
            'clause': "Supplier shall deliver goods within 10 business days.",
            'context': {"Supplier": "CompanyX"}
        },
        {
            'clause': "Either party may terminate with 30 days written notice.",
            'context': {"Party": "PartyA"}
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Clause: {test['clause']}")
        print(f"Context: {test['context']}")
        
        result = generator.generate_with_validation(
            test['clause'],
            test['context']
        )
        
        print(f"FOPL: {result['fopl']}")
        print(f"Valid: {result['validation']['valid']}")
        if not result['validation']['valid']:
            print(f"Issues: {result['validation']['issues']}")
    
    print("\n" + "="*60 + "\n")
