"""
Simplified Compliance Checker
Rule-based compliance checking for common contract scenarios.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from cuad_integration.clause_extractor import CUADClauseExtractor
from cuad_integration.action_parser import ActionParser, ParsedAction, ActionType


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
    parsed_action: Optional[ParsedAction]
    breach_details: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "relevant_clauses": self.relevant_clauses,
            "parsed_action": str(self.parsed_action) if self.parsed_action else None,
            "breach_details": self.breach_details
        }


class SimpleComplianceChecker:
    """Simplified compliance checker using rule-based reasoning."""
    
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
        ActionType.PURCHASE: ["Minimum Commitment", "Volume Restriction"],
    }
    
    def __init__(self, cuad_model_path: str):
        """
        Initialize compliance checker.
        
        Args:
            cuad_model_path: Path to CUAD extraction model
        """
        print("Initializing Simple Compliance Checker...")
        self.clause_extractor = CUADClauseExtractor(cuad_model_path)
        self.action_parser = ActionParser()
        print("✓ Compliance Checker initialized\n")
    
    def check_compliance(
        self,
        contract_text: str,
        user_query: str
    ) -> ComplianceResult:
        """
        Check if an action complies with contract.
        
        Args:
            contract_text: Full contract text
            user_query: User's compliance question
            
        Returns:
            ComplianceResult with status and explanation
        """
        print(f"\n{'='*80}")
        print("CHECKING COMPLIANCE")
        print(f"{'='*80}\n")
        
        # Step 1: Parse user query
        print("📝 Step 1: Parsing user query...")
        print(f"   Query: '{user_query}'")
        parsed_action = self.action_parser.parse(user_query)
        
        if not parsed_action:
            return ComplianceResult(
                status=ComplianceStatus.UNCERTAIN,
                confidence=0.0,
                explanation="Could not parse the action from your query. Please rephrase.",
                relevant_clauses=[],
                parsed_action=None
            )
        
        print(f"   ✓ Parsed: {parsed_action.action_type.value.upper()}")
        print(f"   ✓ Actor: {parsed_action.actor}")
        print(f"   ✓ Parameters: {parsed_action.parameters}\n")
        
        # Step 2: Extract relevant clauses
        print("📄 Step 2: Extracting relevant clauses...")
        relevant_categories = self.ACTION_TO_CATEGORIES.get(
            parsed_action.action_type,
            []
        )
        
        if not relevant_categories:
            relevant_categories = list(self.clause_extractor.EXTRACTION_CATEGORIES.keys())[:5]
        
        extracted_clauses = self.clause_extractor.extract_and_format(contract_text)
        
        # Filter to relevant categories
        relevant_clauses = []
        for category in relevant_categories:
            if category in extracted_clauses and extracted_clauses[category] != "Not found":
                relevant_clauses.append({
                    "category": category,
                    "text": extracted_clauses[category],
                    "confidence": 0.8
                })
                print(f"   ✓ Found: {category}")
        
        if not relevant_clauses:
            print("   ⚠️  No relevant clauses found")
            return ComplianceResult(
                status=ComplianceStatus.INSUFFICIENT_INFO,
                confidence=0.0,
                explanation="Could not find relevant clauses in the contract for this action.",
                relevant_clauses=[],
                parsed_action=parsed_action
            )
        
        print(f"   ✓ Extracted {len(relevant_clauses)} relevant clauses\n")
        
        # Step 3: Rule-based compliance checking
        print("⚖️  Step 3: Checking compliance against rules...")
        status, breach_details, confidence = self._check_rules(
            parsed_action,
            relevant_clauses,
            contract_text
        )
        
        print(f"   ✓ Status: {status.value.upper()}")
        print(f"   ✓ Confidence: {confidence:.0%}\n")
        
        # Step 4: Generate explanation
        print("📋 Step 4: Generating explanation...")
        explanation = self._generate_explanation(
            status,
            parsed_action,
            relevant_clauses,
            breach_details
        )
        
        result = ComplianceResult(
            status=status,
            confidence=confidence,
            explanation=explanation,
            relevant_clauses=relevant_clauses,
            parsed_action=parsed_action,
            breach_details=breach_details
        )
        
        print(f"\n{'='*80}")
        print(f"RESULT: {status.value.upper()} ({confidence:.0%} confidence)")
        print(f"{'='*80}\n")
        
        return result
    
    def _check_rules(
        self,
        action: ParsedAction,
        clauses: List[Dict],
        contract_text: str
    ) -> Tuple[ComplianceStatus, Optional[Dict], float]:
        """Rule-based compliance checking."""
        
        action_type = action.action_type
        params = action.parameters
        
        # DELIVERY rules
        if action_type == ActionType.DELIVER:
            return self._check_delivery(params, contract_text)
        
        # PAYMENT rules
        elif action_type == ActionType.PAY:
            return self._check_payment(params, contract_text)
        
        # TERMINATION rules
        elif action_type == ActionType.TERMINATE:
            return self._check_termination(params, contract_text)
        
        # PURCHASE rules
        elif action_type == ActionType.PURCHASE:
            return self._check_purchase(params, contract_text)
        
        # Default: uncertain
        return ComplianceStatus.UNCERTAIN, None, 0.5
    
    def _check_delivery(self, params: Dict, contract_text: str) -> Tuple[ComplianceStatus, Optional[Dict], float]:
        """Check delivery compliance."""
        
        # Extract delivery timeframe from contract - try multiple patterns
        patterns = [
            r'deliver[^.]*?within\s+([a-z]+\s+)?\((\d+)\)',  # "within ten (10)"
            r'deliver[^.]*?within\s+(\d+)\s+(business\s+)?days?',  # "within 10 days"
            r'deliver[^.]*?(\d+)\s+(business\s+)?days?',  # "10 business days"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, contract_text, re.IGNORECASE)
            if match:
                # Get the numeric value (could be in group 1 or 2)
                required_days = None
                for group in match.groups():
                    if group and group.strip().isdigit():
                        required_days = int(group.strip())
                        break
                
                if required_days:
                    actual_days = params.get('days', 0)
                    
                    if actual_days > required_days:
                        return ComplianceStatus.BREACH, {
                            "required": f"{required_days} business days",
                            "actual": f"{actual_days} days",
                            "difference": actual_days - required_days
                        }, 0.85
                    else:
                        return ComplianceStatus.COMPLIANT, {
                            "required": f"{required_days} business days",
                            "actual": f"{actual_days} days"
                        }, 0.85
        
        return ComplianceStatus.UNCERTAIN, None, 0.3
    
    def _check_payment(self, params: Dict, contract_text: str) -> Tuple[ComplianceStatus, Optional[Dict], float]:
        """Check payment compliance."""
        
        # Extract payment terms from contract - try multiple patterns
        patterns = [
            r'pay[^.]*?within\s+([a-z]+\s+)?\((\d+)\)',  # "within thirty (30)"
            r'pay[^.]*?within\s+(\d+)\s+days?',  # "within 30 days"
            r'pay[^.]*?(\d+)\s+days?',  # "30 days"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, contract_text, re.IGNORECASE)
            if match:
                # Get the numeric value
                required_days = None
                for group in match.groups():
                    if group and group.strip().isdigit():
                        required_days = int(group.strip())
                        break
                
                if required_days:
                    actual_day = params.get('day', 0)
                    
                    if actual_day > required_days:
                        return ComplianceStatus.BREACH, {
                            "required": f"within {required_days} days",
                            "actual": f"day {actual_day}",
                            "difference": actual_day - required_days
                        }, 0.85
                    else:
                        return ComplianceStatus.COMPLIANT, {
                            "required": f"within {required_days} days",
                            "actual": f"day {actual_day}"
                        }, 0.85
        
        return ComplianceStatus.UNCERTAIN, None, 0.3
    
    def _check_termination(self, params: Dict, contract_text: str) -> Tuple[ComplianceStatus, Optional[Dict], float]:
        """Check termination compliance."""
        
        # Extract termination notice requirement from contract - try multiple patterns
        patterns = [
            r'terminat[^.]*?(?:upon providing|with)\s+([a-z]+\s+)?\((\d+)\)\s+days?[^.]*?notice',  # "upon providing ninety (90) days"
            r'terminat[^.]*?(?:upon|with)\s+(\d+)\s+days?[^.]*?notice',  # "upon 90 days notice"
            r'terminat[^.]*?(\d+)\s+days?[^.]*?(?:prior|written)?\s*notice',  # "90 days notice"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, contract_text, re.IGNORECASE)
            if match:
                # Get the numeric value
                required_days = None
                for group in match.groups():
                    if group and group.strip().isdigit():
                        required_days = int(group.strip())
                        break
                
                if required_days:
                    actual_days = params.get('notice_days', 0)
                    
                    if actual_days < required_days:
                        return ComplianceStatus.BREACH, {
                            "required": f"{required_days} days notice",
                            "actual": f"{actual_days} days notice",
                            "difference": required_days - actual_days
                        }, 0.85
                    else:
                        return ComplianceStatus.COMPLIANT, {
                            "required": f"{required_days} days notice",
                            "actual": f"{actual_days} days notice"
                        }, 0.85
        
        return ComplianceStatus.UNCERTAIN, None, 0.3
    
    def _check_purchase(self, params: Dict, contract_text: str) -> Tuple[ComplianceStatus, Optional[Dict], float]:
        """Check purchase/minimum commitment compliance."""
        
        # Extract minimum commitment from contract - try multiple patterns
        patterns = [
            r'(?:purchase|buy)[^.]*?minimum[^.]*?([a-z]+\s+)?\((\d+(?:,\d+)?)\)\s+units?',  # "minimum of one thousand (1,000) units"
            r'(?:purchase|buy)[^.]*?minimum[^.]*?(\d+(?:,\d+)?)\s+units?',  # "minimum 1000 units"
            r'minimum.*?(\d+(?:,\d+)?)\s+units?',  # "minimum 1000 units"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, contract_text, re.IGNORECASE)
            if match:
                # Get the numeric value
                required_units = None
                for group in match.groups():
                    if group and re.search(r'\d', group):
                        required_units = int(re.sub(r'[^\d]', '', group))
                        break
                
                if required_units:
                    actual_units = params.get('quantity', 0)
                    
                    if actual_units < required_units:
                        return ComplianceStatus.BREACH, {
                            "required": f"{required_units} units minimum",
                            "actual": f"{actual_units} units",
                            "difference": required_units - actual_units
                        }, 0.85
                    else:
                        return ComplianceStatus.COMPLIANT, {
                            "required": f"{required_units} units minimum",
                            "actual": f"{actual_units} units"
                        }, 0.85
        
        return ComplianceStatus.UNCERTAIN, None, 0.3
    
    def _generate_explanation(
        self,
        status: ComplianceStatus,
        action: ParsedAction,
        clauses: List[Dict],
        breach_details: Optional[Dict]
    ) -> str:
        """Generate human-readable explanation."""
        
        actor = action.actor
        action_type = action.action_type.value
        
        if status == ComplianceStatus.BREACH:
            if breach_details:
                req = breach_details.get('required', 'requirement')
                act = breach_details.get('actual', 'action')
                diff = breach_details.get('difference', 0)
                
                explanation = f"⚠️ BREACH DETECTED\n\n"
                explanation += f"{actor}'s {action_type} action violates the contract terms.\n\n"
                explanation += f"Contract Requirement: {req}\n"
                explanation += f"Actual Action: {act}\n"
                explanation += f"Violation: Exceeded/Below by {diff}\n\n"
                explanation += f"Relevant clauses:\n"
                for clause in clauses[:2]:
                    explanation += f"  • {clause['category']}: {clause['text'][:100]}...\n"
            else:
                explanation = f"⚠️ BREACH DETECTED\n\n{actor}'s {action_type} action appears to violate contract terms."
        
        elif status == ComplianceStatus.COMPLIANT:
            if breach_details:
                req = breach_details.get('required', 'requirement')
                act = breach_details.get('actual', 'action')
                
                explanation = f"✅ COMPLIANT\n\n"
                explanation += f"{actor}'s {action_type} action complies with the contract terms.\n\n"
                explanation += f"Contract Requirement: {req}\n"
                explanation += f"Actual Action: {act}\n"
            else:
                explanation = f"✅ COMPLIANT\n\n{actor}'s {action_type} action appears to comply with contract terms."
        
        elif status == ComplianceStatus.INSUFFICIENT_INFO:
            explanation = f"ℹ️ INSUFFICIENT INFORMATION\n\n"
            explanation += f"Could not find relevant contract clauses to evaluate {actor}'s {action_type} action."
        
        else:  # UNCERTAIN
            explanation = f"❓ UNCERTAIN\n\n"
            explanation += f"Cannot definitively determine if {actor}'s {action_type} action complies.\n"
            explanation += f"Please review the relevant clauses manually."
        
        return explanation


# For backward compatibility
ComplianceChecker = SimpleComplianceChecker
