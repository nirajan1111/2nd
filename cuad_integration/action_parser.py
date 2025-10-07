"""
Action Parser
Parses user queries into structured actions for compliance checking.

Example:
    "Supplier delivered goods 15 days after purchase order"
    → Deliver(Supplier, days=15, trigger="purchase_order")
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of actions that can be performed."""
    DELIVER = "deliver"
    PAY = "pay"
    TERMINATE = "terminate"
    NOTIFY = "notify"
    RENEW = "renew"
    PROVIDE = "provide"
    MAINTAIN = "maintain"
    AUDIT = "audit"
    SELL = "sell"
    LICENSE = "license"
    ASSIGN = "assign"
    COMPETE = "compete"


@dataclass
class ParsedAction:
    """Structured representation of a parsed action."""
    action_type: ActionType
    actor: str  # Who performed the action
    parameters: Dict[str, any]  # Action parameters
    raw_text: str  # Original text
    confidence: float  # Parsing confidence
    
    def to_fopl_atom(self) -> str:
        """Convert to FOPL atom format."""
        action_name = self.action_type.value.capitalize()
        
        # Build parameter string
        params = [f"{k}={v}" for k, v in self.parameters.items()]
        param_str = ", ".join(params) if params else ""
        
        if param_str:
            return f"{action_name}({self.actor}, {param_str})"
        else:
            return f"{action_name}({self.actor})"
    
    def __repr__(self):
        return f"ParsedAction({self.action_type.value}, {self.actor}, {self.parameters})"


class ActionParser:
    """Parse natural language action descriptions into structured format."""
    
    # Patterns for different action types
    ACTION_PATTERNS = {
        ActionType.DELIVER: [
            (r"(\w+)\s+delivered\s+(?:goods|product|items)?\s*(?:in|within|after)?\s*(\d+)\s*(days?|weeks?|months?)",
             lambda m: {"days": int(m.group(2)) * (30 if 'month' in m.group(3) else 7 if 'week' in m.group(3) else 1)}),
            
            (r"(\w+)\s+(?:failed to deliver|did not deliver)",
             lambda m: {"status": "not_delivered"}),
        ],
        
        ActionType.PAY: [
            (r"(\w+)\s+paid\s+(?:on|by|in)?\s*(?:the\s*)?(\d+)(?:th|st|nd|rd)?\s*(day|month)?",
             lambda m: {"day": int(m.group(2))}),
            
            (r"(\w+)\s+paid\s+\$?(\d+(?:,\d+)?(?:\.\d+)?)",
             lambda m: {"amount": float(m.group(2).replace(',', ''))}),
            
            (r"(\w+)\s+(?:failed to pay|did not pay)",
             lambda m: {"status": "not_paid"}),
        ],
        
        ActionType.TERMINATE: [
            (r"(\w+)\s+terminated\s+(?:the\s+)?(?:contract|agreement)\s*(?:with)?\s*(\d+)?\s*(days?|months?)?\s*(?:notice)?",
             lambda m: {"notice_days": int(m.group(2)) if m.group(2) else 0}),
            
            (r"(\w+)\s+terminated\s+(?:without|with no)\s+notice",
             lambda m: {"notice_days": 0}),
        ],
        
        ActionType.NOTIFY: [
            (r"(\w+)\s+(?:notified|gave notice)\s+(\d+)\s*(days?|weeks?|months?)\s*(?:before|prior|in advance)",
             lambda m: {"notice_days": int(m.group(2)) * (30 if 'month' in m.group(3) else 7 if 'week' in m.group(3) else 1)}),
            
            (r"(\w+)\s+(?:did not notify|failed to notify)",
             lambda m: {"status": "not_notified"}),
        ],
        
        ActionType.SELL: [
            (r"(\w+)\s+sold\s+(?:products?|goods?)\s+(?:to|directly to)\s+(\w+)",
             lambda m: {"customer": m.group(2)}),
            
            (r"(\w+)\s+sold\s+directly",
             lambda m: {"direct_sale": True}),
        ],
        
        ActionType.RENEW: [
            (r"(\w+)\s+(?:renewed|extended)\s+(?:the\s+)?(?:contract|agreement)",
             lambda m: {"renewed": True}),
            
            (r"(\w+)\s+(?:did not renew|failed to renew)",
             lambda m: {"renewed": False}),
        ],
        
        ActionType.MAINTAIN: [
            (r"(\w+)\s+maintained\s+(?:for|during)?\s*(\d+)\s*(years?|months?)",
             lambda m: {"duration_years": int(m.group(2)) / (12 if 'month' in m.group(3) else 1)}),
            
            (r"(\w+)\s+stopped\s+maintain(?:ing)?",
             lambda m: {"status": "not_maintained"}),
        ],
        
        ActionType.COMPETE: [
            (r"(\w+)\s+(?:competed with|is competing with)\s+(\w+)",
             lambda m: {"competitor": m.group(2)}),
            
            (r"(\w+)\s+entered\s+(?:the\s+)?market",
             lambda m: {"market_entry": True}),
        ],
    }
    
    # Entity recognition patterns
    ENTITY_PATTERNS = [
        (r"\b(Supplier|Buyer|Vendor|Customer|Client|Provider|Contractor|Party[AB]?|Licensor|Licensee)\b", "role"),
        (r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s+Inc\.?|\s+LLC|\s+Corp\.?)?)\b", "company"),
    ]
    
    # Temporal expressions
    TEMPORAL_PATTERNS = [
        (r"(\d+)\s*(days?|weeks?|months?|years?)\s*(?:late|after|early|before)", 
         lambda m: int(m.group(1)) * (365 if 'year' in m.group(2) else 30 if 'month' in m.group(2) else 7 if 'week' in m.group(2) else 1)),
    ]
    
    def __init__(self):
        """Initialize action parser."""
        pass
    
    def parse(self, text: str) -> Optional[ParsedAction]:
        """
        Parse action description into structured format.
        
        Args:
            text: Natural language action description
            
        Returns:
            ParsedAction if successfully parsed, None otherwise
        """
        text = text.strip().lower()
        
        # Try each action type
        for action_type, patterns in self.ACTION_PATTERNS.items():
            for pattern, param_extractor in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Extract actor
                    actor = match.group(1)
                    
                    # Extract parameters
                    try:
                        parameters = param_extractor(match)
                    except Exception as e:
                        print(f"Warning: Failed to extract parameters: {e}")
                        parameters = {}
                    
                    # Add temporal information if present
                    temporal_info = self._extract_temporal(text)
                    if temporal_info:
                        parameters.update(temporal_info)
                    
                    return ParsedAction(
                        action_type=action_type,
                        actor=actor.capitalize(),
                        parameters=parameters,
                        raw_text=text,
                        confidence=0.9  # High confidence for pattern match
                    )
        
        # No pattern matched
        return None
    
    def parse_multiple(self, text: str) -> List[ParsedAction]:
        """Parse multiple actions from text (separated by commas, 'and', etc.)"""
        
        # Split on common separators
        sentences = re.split(r'[,;]|\band\b', text)
        
        actions = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short fragments
                action = self.parse(sentence)
                if action:
                    actions.append(action)
        
        return actions
    
    def _extract_temporal(self, text: str) -> Dict[str, int]:
        """Extract temporal information from text."""
        temporal = {}
        
        for pattern, extractor in self.TEMPORAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if 'late' in text or 'after' in text:
                    temporal['delay_days'] = extractor(match)
                elif 'early' in text or 'before' in text:
                    temporal['early_days'] = extractor(match)
        
        return temporal
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        entities = {"roles": [], "companies": []}
        
        for pattern, entity_type in self.ENTITY_PATTERNS:
            matches = re.findall(pattern, text)
            if entity_type == "role":
                entities["roles"].extend(matches)
            elif entity_type == "company":
                entities["companies"].extend(matches)
        
        return entities


def test_parser():
    """Test the action parser."""
    
    parser = ActionParser()
    
    test_cases = [
        "Supplier delivered goods 15 days after purchase order",
        "Tenant paid rent on the 8th day",
        "Buyer terminated the contract with 30 days notice",
        "Vendor did not notify before termination",
        "Contractor maintained for 3 years",
        "Party A sold products directly to CustomerX",
        "Licensee failed to pay licensing fees",
        "Provider competed with the other party",
    ]
    
    print("\n" + "="*70)
    print("Action Parser Tests")
    print("="*70 + "\n")
    
    for text in test_cases:
        print(f"Input: {text}")
        action = parser.parse(text)
        
        if action:
            print(f"  Action Type: {action.action_type.value}")
            print(f"  Actor: {action.actor}")
            print(f"  Parameters: {action.parameters}")
            print(f"  FOPL: {action.to_fopl_atom()}")
        else:
            print("  ✗ Failed to parse")
        
        print()
    
    print("="*70)


if __name__ == '__main__':
    test_parser()
