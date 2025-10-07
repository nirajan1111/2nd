"""
Compliance Checker
Main module for checking contract compliance using extracted clauses and FOPL rules.

Usage:
    checker = ComplianceChecker(
        cuad_model_path="...",
        fopl_model_path="..."
    )
    
    result = checker.check_compliance(
        contract_text="...",
        user_query="Supplier delivered 15 days late, is this breach?"
    )
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from cuad_integration.clause_extractor import CUADClauseExtractor
from cuad_integration.action_parser import ActionParser, ParsedAction, ActionType
from models.neural_parser import NeuralLegalParser
from models.symbolic_reasoner import SymbolicReasoner
from inference.explainer import LegalExplainer


class ComplianceStatus(Enum):
    """Compliance status outcomes."""
    COMPLIANT = "compliant"
    BREACH = "breach"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_INFO = "insufficient_information"


@dataclass
class ComplianceResult:
    """Result of compliance checking."""
    status: ComplianceStatus
    confidence: float
    explanation: str
    relevant_clauses: List[Dict]
    fopl_rules: List[str]
    parsed_action: Optional[ParsedAction]
    breach_details: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "relevant_clauses": self.relevant_clauses,
            "fopl_rules": self.fopl_rules,
            "parsed_action": str(self.parsed_action) if self.parsed_action else None,
            "breach_details": self.breach_details
        }


class ComplianceChecker:
    """Check contract compliance by combining clause extraction, FOPL generation, and symbolic reasoning."""
    
    # Map action types to relevant CUAD categories
    ACTION_TO_CATEGORIES = {
        ActionType.DELIVER: ["Minimum Commitment", "Volume Restriction"],
        ActionType.PAY: ["Revenue/Profit Sharing", "Price Restrictions"],
        ActionType.TERMINATE: ["Termination For Convenience", "Notice Period To Terminate Renewal"],
        ActionType.NOTIFY: ["Notice Period To Terminate Renewal"],
        ActionType.SELL: ["Exclusivity", "Non-Compete"],
        ActionType.COMPETE: ["Non-Compete", "Exclusivity"],
        ActionType.MAINTAIN: ["Warranty Duration"],
        ActionType.AUDIT: ["Audit Rights"],
        ActionType.RENEW: ["Renewal Term", "Expiration Date"],
    }
    
    def __init__(
        self,
        cuad_model_path: str,
        fopl_model_path: str,
        enable_explainer: bool = True
    ):
        """
        Initialize compliance checker.
        
        Args:
            cuad_model_path: Path to trained CUAD extraction model
            fopl_model_path: Path to trained FOPL generation model
            enable_explainer: Whether to generate detailed explanations
        """
        print("Initializing Compliance Checker...")
        
        # Initialize models
        self.clause_extractor = CUADClauseExtractor(cuad_model_path)
        self.fopl_generator = NeuralLegalParser(model_path=fopl_model_path)
        self.action_parser = ActionParser()
        self.reasoner = SymbolicReasoner()
        
        if enable_explainer:
            self.explainer = LegalExplainer()
        else:
            self.explainer = None
        
        print("✓ Compliance Checker initialized")
    
    def check_compliance(
        self,
        contract_text: str,
        user_query: str,
        parties: Optional[Dict[str, str]] = None
    ) -> ComplianceResult:
        """
        Check if an action complies with contract.
        
        Args:
            contract_text: Full contract text
            user_query: User's compliance question (e.g., "Supplier delivered 15 days late")
            parties: Optional mapping of role names to actual party names
            
        Returns:
            ComplianceResult with status and explanation
        """
        print(f"\n{'='*70}")
        print("Checking Compliance")
        print(f"{'='*70}\n")
        
        # Step 1: Parse user query to extract action
        print("Step 1: Parsing user query...")
        parsed_action = self.action_parser.parse(user_query)
        
        if not parsed_action:
            return ComplianceResult(
                status=ComplianceStatus.UNCERTAIN,
                confidence=0.0,
                explanation="Could not parse the action from your query. Please rephrase.",
                relevant_clauses=[],
                fopl_rules=[],
                parsed_action=None
            )
        
        print(f"  ✓ Parsed action: {parsed_action.to_fopl_atom()}")
        
        # Step 2: Extract relevant clauses from contract
        print("\nStep 2: Extracting relevant clauses from contract...")
        relevant_categories = self.ACTION_TO_CATEGORIES.get(
            parsed_action.action_type,
            list(CUADClauseExtractor.EXTRACTION_CATEGORIES.keys())
        )
        
        # Also always extract parties
        if "Parties" not in relevant_categories:
            relevant_categories = ["Parties"] + relevant_categories
        
        extracted_clauses = self.clause_extractor.extract_clauses(
            contract_text,
            categories=relevant_categories,
            confidence_threshold=0.3
        )
        
        print(f"  ✓ Extracted {len(extracted_clauses)} categories")
        for category in extracted_clauses.keys():
            print(f"    - {category}")
        
        # Extract parties if not provided
        if not parties and "Parties" in extracted_clauses:
            parties = self._extract_party_names(extracted_clauses["Parties"])
            print(f"  ✓ Identified parties: {parties}")
        
        # Step 3: Generate FOPL rules from extracted clauses
        print("\nStep 3: Generating FOPL rules...")
        fopl_rules = []
        relevant_clauses = []
        
        for category, extractions in extracted_clauses.items():
            if category == "Parties":
                continue  # Skip parties category for FOPL generation
            
            for extraction in extractions[:1]:  # Use top extraction
                clause_text = extraction['text']
                
                # Generate FOPL
                try:
                    fopl = self.fopl_generator.parse(clause_text)
                    if fopl and len(fopl.strip()) > 0:
                        fopl_rules.append(fopl)
                        relevant_clauses.append({
                            "category": category,
                            "text": clause_text,
                            "fopl": fopl,
                            "confidence": extraction['confidence']
                        })
                        print(f"  ✓ {category}:")
                        print(f"      Clause: {clause_text[:100]}...")
                        print(f"      FOPL: {fopl}")
                except Exception as e:
                    print(f"  ✗ Failed to generate FOPL for {category}: {e}")
        
        if not fopl_rules:
            return ComplianceResult(
                status=ComplianceStatus.INSUFFICIENT_INFO,
                confidence=0.0,
                explanation="Could not find relevant clauses in the contract for this action.",
                relevant_clauses=[],
                fopl_rules=[],
                parsed_action=parsed_action
            )
        
        # Step 4: Symbolic reasoning
        print("\nStep 4: Checking compliance...")
        compliance_status, breach_details = self._check_against_rules(
            parsed_action,
            fopl_rules,
            relevant_clauses
        )
        
        print(f"  ✓ Status: {compliance_status.value}")
        
        # Step 5: Generate explanation
        print("\nStep 5: Generating explanation...")
        explanation = self._generate_explanation(
            compliance_status,
            parsed_action,
            relevant_clauses,
            breach_details,
            parties
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            parsed_action,
            relevant_clauses,
            compliance_status
        )
        
        result = ComplianceResult(
            status=compliance_status,
            confidence=confidence,
            explanation=explanation,
            relevant_clauses=relevant_clauses,
            fopl_rules=fopl_rules,
            parsed_action=parsed_action,
            breach_details=breach_details
        )
        
        print(f"\n{'='*70}")
        print(f"Result: {compliance_status.value.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"{'='*70}\n")
        
        return result
    
    def _extract_party_names(self, party_extractions: List[Dict]) -> Dict[str, str]:
        """Extract party names from Parties clause."""
        parties = {}
        
        if party_extractions:
            text = party_extractions[0]['text']
            
            # Simple extraction - look for patterns like "Company A" and "Company B"
            import re
            
            # Look for quoted names or capitalized entities
            quoted = re.findall(r'"([^"]+)"', text)
            if len(quoted) >= 2:
                parties = {
                    "Party A": quoted[0],
                    "Party B": quoted[1]
                }
            
            # Look for "between X and Y" pattern
            between_pattern = r'between\s+([A-Z][A-Za-z\s,\.]+?)\s+(?:and|&)\s+([A-Z][A-Za-z\s,\.]+?)(?:\s|$|\.)'
            match = re.search(between_pattern, text)
            if match:
                parties = {
                    "Party A": match.group(1).strip(),
                    "Party B": match.group(2).strip()
                }
        
        return parties
    
    def _check_against_rules(
        self,
        action: ParsedAction,
        fopl_rules: List[str],
        relevant_clauses: List[Dict]
    ) -> Tuple[ComplianceStatus, Optional[Dict]]:
        """Check action against FOPL rules using symbolic reasoning."""
        
        # Extract constraint from relevant clauses
        constraint_violated = None
        
        for clause in relevant_clauses:
            fopl = clause['fopl']
            category = clause['category']
            
            # Check for violations based on action type and parameters
            violation = self._check_constraint_violation(action, fopl, clause)
            
            if violation:
                constraint_violated = violation
                return ComplianceStatus.BREACH, violation
        
        # If no violations found, assume compliant
        return ComplianceStatus.COMPLIANT, None
    
    def _check_constraint_violation(
        self,
        action: ParsedAction,
        fopl: str,
        clause: Dict
    ) -> Optional[Dict]:
        """Check if action violates a specific constraint."""
        
        # Extract numeric constraints from FOPL
        import re
        
        # Look for patterns like "days <= 10" or "notice_days >= 30"
        constraint_patterns = [
            (r'(\w+)\s*<=\s*(\d+)', 'max'),
            (r'(\w+)\s*>=\s*(\d+)', 'min'),
            (r'(\w+)\s*<\s*(\d+)', 'max_exclusive'),
            (r'(\w+)\s*>\s*(\d+)', 'min_exclusive'),
            (r'(\w+)\s*=\s*(\d+)', 'exact'),
        ]
        
        for pattern, constraint_type in constraint_patterns:
            match = re.search(pattern, fopl)
            if match:
                param_name = match.group(1)
                threshold = int(match.group(2))
                
                # Check if action has this parameter
                action_value = None
                for key in action.parameters:
                    if key in param_name or param_name in key:
                        action_value = action.parameters[key]
                        break
                
                if action_value is not None:
                    # Check violation
                    is_violated = False
                    
                    if constraint_type == 'max' and action_value > threshold:
                        is_violated = True
                    elif constraint_type == 'min' and action_value < threshold:
                        is_violated = True
                    elif constraint_type == 'max_exclusive' and action_value >= threshold:
                        is_violated = True
                    elif constraint_type == 'min_exclusive' and action_value <= threshold:
                        is_violated = True
                    elif constraint_type == 'exact' and action_value != threshold:
                        is_violated = True
                    
                    if is_violated:
                        return {
                            "parameter": param_name,
                            "constraint_type": constraint_type,
                            "threshold": threshold,
                            "actual_value": action_value,
                            "clause_category": clause['category'],
                            "clause_text": clause['text'],
                            "fopl": fopl
                        }
        
        return None
    
    def _generate_explanation(
        self,
        status: ComplianceStatus,
        action: ParsedAction,
        clauses: List[Dict],
        breach_details: Optional[Dict],
        parties: Optional[Dict]
    ) -> str:
        """Generate human-readable explanation."""
        
        if status == ComplianceStatus.BREACH and breach_details:
            explanation = f"⚠️ BREACH DETECTED\n\n"
            explanation += f"The action violates the contract's {breach_details['clause_category']} clause.\n\n"
            explanation += f"**Contract Requirement:**\n"
            explanation += f'"{breach_details["clause_text"][:200]}..."\n\n'
            explanation += f"**Formal Logic:**\n"
            explanation += f"`{breach_details['fopl']}`\n\n"
            explanation += f"**What Happened:**\n"
            explanation += f"{action.actor} performed: {action.raw_text}\n\n"
            explanation += f"**Violation:**\n"
            explanation += f"The {breach_details['parameter']} was {breach_details['actual_value']}, "
            explanation += f"but the contract requires {breach_details['constraint_type']} {breach_details['threshold']}.\n\n"
            
            # Calculate severity
            if breach_details['constraint_type'] in ['max', 'max_exclusive']:
                excess = breach_details['actual_value'] - breach_details['threshold']
                explanation += f"This exceeds the limit by {excess} {breach_details['parameter']}.\n\n"
            elif breach_details['constraint_type'] in ['min', 'min_exclusive']:
                shortfall = breach_details['threshold'] - breach_details['actual_value']
                explanation += f"This falls short by {shortfall} {breach_details['parameter']}.\n\n"
            
            explanation += f"**Consequences:**\n"
            explanation += f"This may entitle the other party to:\n"
            explanation += f"- Claim damages for breach\n"
            explanation += f"- Terminate the contract (if material breach)\n"
            explanation += f"- Seek specific performance\n"
        
        elif status == ComplianceStatus.COMPLIANT:
            explanation = f"✓ COMPLIANT\n\n"
            explanation += f"The action complies with all relevant contract terms.\n\n"
            explanation += f"**Action Performed:**\n"
            explanation += f"{action.actor}: {action.raw_text}\n\n"
            explanation += f"**Relevant Clauses Checked:**\n"
            for clause in clauses:
                explanation += f"- {clause['category']}: ✓ Compliant\n"
        
        else:
            explanation = f"❓ UNCERTAIN\n\n"
            explanation += f"Could not determine compliance status definitively.\n"
        
        return explanation
    
    def _calculate_confidence(
        self,
        action: ParsedAction,
        clauses: List[Dict],
        status: ComplianceStatus
    ) -> float:
        """Calculate confidence score for compliance result."""
        
        # Base confidence from action parsing
        confidence = action.confidence
        
        # Boost if we have high-confidence clause extractions
        if clauses:
            avg_clause_confidence = sum(c['confidence'] for c in clauses) / len(clauses)
            confidence = (confidence + avg_clause_confidence) / 2
        
        # Reduce if status is uncertain
        if status == ComplianceStatus.UNCERTAIN:
            confidence *= 0.5
        elif status == ComplianceStatus.INSUFFICIENT_INFO:
            confidence *= 0.3
        
        return min(confidence, 1.0)


def test_compliance_checker():
    """Test the compliance checker."""
    
    # Sample contract
    contract_text = """
    SUPPLY AGREEMENT
    
    This Agreement is between SupplierCorp ("Supplier") and BuyerInc ("Buyer").
    
    1. DELIVERY OBLIGATIONS
    Supplier shall deliver all goods within 10 business days of receiving a 
    purchase order from Buyer.
    
    2. PAYMENT TERMS
    Buyer shall pay all invoices within 30 days of receipt.
    
    3. EXCLUSIVITY
    All sales of products covered by this agreement shall be conducted 
    exclusively by Supplier. Buyer shall not sell directly to end customers.
    
    4. GOVERNING LAW
    This Agreement shall be governed by the laws of the State of Delaware.
    """
    
    # Test queries
    test_queries = [
        "Supplier delivered goods 15 days after purchase order",
        "Buyer paid invoice 25 days after receipt",
        "Buyer sold products directly to CustomerX",
    ]
    
    # Initialize checker (assumes models are trained)
    cuad_model = "../cuad_models/roberta_extractor/best_model"
    fopl_model = "../checkpoints/best_model"
    
    if not Path(cuad_model).exists():
        print("CUAD model not found. Please train it first.")
        return
    
    if not Path(fopl_model).exists():
        print("FOPL model not found. Please train it first.")
        return
    
    checker = ComplianceChecker(cuad_model, fopl_model)
    
    # Check compliance for each query
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        result = checker.check_compliance(contract_text, query)
        
        print(f"\nResult:")
        print(result.explanation)


if __name__ == '____':
    test_compliance_checker()
