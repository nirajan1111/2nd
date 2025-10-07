"""
CUAD Clause Extractor
Production-ready wrapper for extracting clauses from contracts.

Usage:
    from cuad_integration.clause_extractor import CUADClauseExtractor
    
    extractor = CUADClauseExtractor("../cuad_models/roberta_extractor/best_model")
    clauses = extractor.extract_clauses(contract_text)
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path


class CUADClauseExtractor:
    """Extract clauses from contracts using trained CUAD model."""
    
    # Define categories to extract (most important for compliance)
    EXTRACTION_CATEGORIES = {
        "Parties": "Highlight the parts (if any) of this contract related to \"Parties\" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract",
        
        "Governing Law": "Highlight the parts (if any) of this contract related to \"Governing Law\" that should be reviewed by a lawyer. Details: Which state/country's law governs the interpretation of the contract?",
        
        "Expiration Date": "Highlight the parts (if any) of this contract related to \"Expiration Date\" that should be reviewed by a lawyer. Details: On what date will the contract's initial term expire?",
        
        "Termination For Convenience": "Highlight the parts (if any) of this contract related to \"Termination For Convenience\" that should be reviewed by a lawyer. Details: Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)?",
        
        "Minimum Commitment": "Highlight the parts (if any) of this contract related to \"Minimum Commitment\" that should be reviewed by a lawyer. Details: Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?",
        
        "Volume Restriction": "Highlight the parts (if any) of this contract related to \"Volume Restriction\" that should be reviewed by a lawyer. Details: Is there a fee increase or consent requirement, etc. if one party's use of the product/services exceeds certain threshold?",
        
        "Exclusivity": "Highlight the parts (if any) of this contract related to \"Exclusivity\" that should be reviewed by a lawyer. Details: Is there an exclusive dealing commitment with the counterparty?",
        
        "Non-Compete": "Highlight the parts (if any) of this contract related to \"Non-Compete\" that should be reviewed by a lawyer. Details: Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?",
        
        "Cap On Liability": "Highlight the parts (if any) of this contract related to \"Cap On Liability\" that should be reviewed by a lawyer. Details: Does the contract include a cap on liability upon the breach of a party's obligation?",
        
        "Warranty Duration": "Highlight the parts (if any) of this contract related to \"Warranty Duration\" that should be reviewed by a lawyer. Details: What is the duration of any  warranty against defects or errors in technology, products, or services  provided  under the contract?",
        
        "Audit Rights": "Highlight the parts (if any) of this contract related to \"Audit Rights\" that should be reviewed by a lawyer. Details: Does a party have the right to  audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?",
        
        "Revenue/Profit Sharing": "Highlight the parts (if any) of this contract related to \"Revenue/Profit Sharing\" that should be reviewed by a lawyer. Details: Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?",
        
        "Price Restrictions": "Highlight the parts (if any) of this contract related to \"Price Restrictions\" that should be reviewed by a lawyer. Details: Is there a restriction on the  ability of a party to raise or reduce prices or fees?",
        
        "Renewal Term": "Highlight the parts (if any) of this contract related to \"Renewal Term\" that should be reviewed by a lawyer. Details: What is the renewal term after the initial term expires?",
        
        "Notice Period To Terminate Renewal": "Highlight the parts (if any) of this contract related to \"Notice Period To Terminate Renewal\" that should be reviewed by a lawyer. Details: What is the notice period required to terminate renewal?",
    }
    
    def __init__(self, model_path: str, max_length: int = 512, doc_stride: int = 128):
        """
        Initialize CUAD clause extractor.
        
        Args:
            model_path: Path to trained model directory
            max_length: Maximum sequence length
            doc_stride: Stride for sliding window
        """
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.doc_stride = doc_stride
        
        # Load model and tokenizer
        print(f"Loading CUAD extraction model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForQuestionAnswering.from_pretrained(str(self.model_path))
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    def extract_clauses(
        self, 
        contract_text: str,
        categories: Optional[List[str]] = None,
        confidence_threshold: float = 0.3
    ) -> Dict[str, List[Dict]]:
        """
        Extract clauses from contract text.
        
        Args:
            contract_text: Full contract text
            categories: List of categories to extract (if None, extract all)
            confidence_threshold: Minimum confidence score to include answer
            
        Returns:
            Dictionary mapping category to list of extracted clause dicts
        """
        if categories is None:
            categories = list(self.EXTRACTION_CATEGORIES.keys())
        
        results = {}
        
        for category in categories:
            if category not in self.EXTRACTION_CATEGORIES:
                print(f"Warning: Unknown category '{category}', skipping")
                continue
            
            question = self.EXTRACTION_CATEGORIES[category]
            extracted = self._extract_single_category(
                contract_text, 
                question, 
                category,
                confidence_threshold
            )
            
            if extracted:
                results[category] = extracted
        
        return results
    
    def _extract_single_category(
        self,
        context: str,
        question: str,
        category: str,
        confidence_threshold: float
    ) -> List[Dict]:
        """Extract answer for a single category."""
        
        # Tokenize with sliding window
        inputs = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        offset_mapping = inputs.pop("offset_mapping")
        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()
        
        # Find best answer across all windows
        best_answers = []
        
        for idx in range(len(start_logits)):
            start_scores = start_logits[idx]
            end_scores = end_logits[idx]
            offsets = offset_mapping[idx]
            
            # Get top-k start and end positions
            top_k = 20
            start_indices = np.argsort(start_scores)[-top_k:]
            end_indices = np.argsort(end_scores)[-top_k:]
            
            for start_idx in start_indices:
                for end_idx in end_indices:
                    # Skip if end before start or too long
                    if end_idx < start_idx or end_idx - start_idx > 100:
                        continue
                    
                    # Skip if answer is in padding or question
                    if offsets[start_idx] is None or offsets[end_idx] is None:
                        continue
                    if offsets[start_idx][0] == 0 and offsets[start_idx][1] == 0:
                        continue
                    
                    # Calculate confidence score
                    score = start_scores[start_idx] + end_scores[end_idx]
                    confidence = self._sigmoid(score)
                    
                    if confidence < confidence_threshold:
                        continue
                    
                    # Extract answer text
                    start_char = offsets[start_idx][0].item()
                    end_char = offsets[end_idx][1].item()
                    answer_text = context[start_char:end_char].strip()
                    
                    if len(answer_text) > 0:
                        best_answers.append({
                            'text': answer_text,
                            'start_char': start_char,
                            'end_char': end_char,
                            'confidence': confidence,
                            'category': category
                        })
        
        # Remove duplicates and sort by confidence
        unique_answers = self._deduplicate_answers(best_answers)
        unique_answers.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_answers[:5]  # Return top 5
    
    def _deduplicate_answers(self, answers: List[Dict], iou_threshold: float = 0.7) -> List[Dict]:
        """Remove duplicate/overlapping answers using IoU."""
        if not answers:
            return []
        
        # Sort by confidence
        answers = sorted(answers, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for answer in answers:
            # Check overlap with kept answers
            is_duplicate = False
            for kept_answer in keep:
                iou = self._compute_iou(
                    (answer['start_char'], answer['end_char']),
                    (kept_answer['start_char'], kept_answer['end_char'])
                )
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(answer)
        
        return keep
    
    def _compute_iou(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
        """Compute intersection over union of two spans."""
        start1, end1 = span1
        start2, end2 = span2
        
        # Intersection
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        intersection = max(0, inter_end - inter_start)
        
        # Union
        union = (end1 - start1) + (end2 - start2) - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))
    
    def extract_and_format(
        self,
        contract_text: str,
        categories: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Extract clauses and return formatted dictionary.
        
        Args:
            contract_text: Full contract text
            categories: Categories to extract
            
        Returns:
            Dict mapping category to best extracted text
        """
        results = self.extract_clauses(contract_text, categories)
        
        formatted = {}
        for category, extractions in results.items():
            if extractions:
                # Take highest confidence extraction
                formatted[category] = extractions[0]['text']
            else:
                formatted[category] = None
        
        return formatted


def test_extractor():
    """Test the clause extractor."""
    
    # Sample contract text
    contract_text = """
    SOFTWARE MAINTENANCE AGREEMENT
    
    This Agreement entered into as of October 13, 2016, by and between 
    Leader Act Ltd ("LEADER") and EZJR, Inc. ("EZJR").
    
    1. TERM
    Subject to all other terms and conditions set forth herein, as of the 
    date of this agreement, LEADER shall maintain the software for an 
    additional five years.
    
    2. GOVERNING LAW
    This Agreement and any matters arising out of or related to this 
    Agreement will be governed by the laws of the State of Nevada.
    
    3. TERMINATION
    Either party may terminate this agreement with 30 days written notice.
    """
    
    # Initialize extractor (assumes model is trained)
    model_path = "../cuad_models/roberta_extractor/best_model"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using train_cuad_extractor.py")
        return
    
    extractor = CUADClauseExtractor(model_path)
    
    # Extract clauses
    print("\nExtracting clauses...")
    results = extractor.extract_and_format(
        contract_text,
        categories=["Parties", "Governing Law", "Expiration Date"]
    )
    
    # Print results
    print("\nExtracted Clauses:")
    print("=" * 70)
    for category, text in results.items():
        print(f"\n{category}:")
        print(f"  {text if text else 'Not found'}")
    print("=" * 70)


if __name__ == '__main__':
    test_extractor()
