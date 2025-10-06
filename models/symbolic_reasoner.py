import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ReasoningResult:
    """Result of symbolic reasoning"""
    outcome: bool
    explanation: str
    proof_trace: List[str]
    satisfied_rules: List[str]
    violated_rules: List[str]


class SymbolicReasoner:
    """
    Symbolic reasoning engine for FOPL
    Simplified Prolog-like inference without external dependencies
    """
    
    def __init__(self):
        self.facts = []
        self.rules = []
        self.predicates = {}
        
    def clear(self):
        """Clear all facts and rules"""
        self.facts = []
        self.rules = []
        self.predicates = {}
    
    def parse_fopl(self, fopl_string: str) -> Dict:
        """
        Parse FOPL string into structured format
        
        Example: "forall x (Tenant(x) -> PayRent(x, due_date <= 5))"
        """
        fopl_string = fopl_string.strip()
        
        # Extract quantifier
        quantifier = None
        if fopl_string.startswith('forall'):
            quantifier = 'forall'
            fopl_string = fopl_string[6:].strip()
        elif fopl_string.startswith('exists'):
            quantifier = 'exists'
            fopl_string = fopl_string[6:].strip()
        
        # Extract variables
        variables = []
        if quantifier:
            var_match = re.match(r'([a-z,\s]+)\s*\(', fopl_string)
            if var_match:
                var_str = var_match.group(1).strip()
                variables = [v.strip() for v in var_str.split(',')]
                # Remove variable declaration
                fopl_string = fopl_string[var_match.end()-1:]
        
        # Parse the body
        body = self._parse_expression(fopl_string)
        
        return {
            'quantifier': quantifier,
            'variables': variables,
            'body': body,
            'original': fopl_string
        }
    
    def _parse_expression(self, expr: str) -> Dict:
        """Parse logical expression"""
        expr = expr.strip()
        
        # Remove outer parentheses
        if expr.startswith('(') and expr.endswith(')'):
            expr = expr[1:-1].strip()
        
        # Check for implication (->)
        if '->' in expr:
            parts = self._split_operator(expr, '->')
            if len(parts) == 2:
                return {
                    'type': 'implication',
                    'antecedent': self._parse_expression(parts[0]),
                    'consequent': self._parse_expression(parts[1])
                }
        
        # Check for conjunction (&)
        if '&' in expr:
            parts = self._split_operator(expr, '&')
            return {
                'type': 'conjunction',
                'operands': [self._parse_expression(p) for p in parts]
            }
        
        # Check for disjunction (|)
        if '|' in expr:
            parts = self._split_operator(expr, '|')
            return {
                'type': 'disjunction',
                'operands': [self._parse_expression(p) for p in parts]
            }
        
        # Check for negation (~)
        if expr.startswith('~'):
            return {
                'type': 'negation',
                'operand': self._parse_expression(expr[1:])
            }
        
        # Parse predicate
        return self._parse_predicate(expr)
    
    def _split_operator(self, expr: str, operator: str) -> List[str]:
        """Split expression by operator, respecting parentheses"""
        parts = []
        current = []
        depth = 0
        i = 0
        
        while i < len(expr):
            if expr[i] == '(':
                depth += 1
                current.append(expr[i])
            elif expr[i] == ')':
                depth -= 1
                current.append(expr[i])
            elif depth == 0 and expr[i:i+len(operator)] == operator:
                parts.append(''.join(current).strip())
                current = []
                i += len(operator) - 1
            else:
                current.append(expr[i])
            i += 1
        
        if current:
            parts.append(''.join(current).strip())
        
        return parts
    
    def _parse_predicate(self, pred_str: str) -> Dict:
        """Parse a predicate like Tenant(x) or PayRent(x, date <= 5)"""
        pred_str = pred_str.strip()
        
        # Extract predicate name and arguments
        match = re.match(r'([A-Z][a-zA-Z]*)\((.*)\)', pred_str)
        if not match:
            return {
                'type': 'predicate',
                'name': pred_str,
                'args': []
            }
        
        name = match.group(1)
        args_str = match.group(2)
        
        # Parse arguments
        args = []
        if args_str:
            for arg in args_str.split(','):
                arg = arg.strip()
                # Check for constraints (<=, >=, =, <, >)
                constraint_match = re.match(r'(\w+)\s*([<>=]+)\s*(.+)', arg)
                if constraint_match:
                    args.append({
                        'var': constraint_match.group(1),
                        'op': constraint_match.group(2),
                        'value': constraint_match.group(3).strip()
                    })
                else:
                    args.append({'var': arg, 'op': None, 'value': None})
        
        return {
            'type': 'predicate',
            'name': name,
            'args': args
        }
    
    def add_rule(self, fopl_string: str):
        """Add a FOPL rule to the knowledge base"""
        parsed = self.parse_fopl(fopl_string)
        self.rules.append(parsed)
    
    def add_fact(self, predicate: str, value: bool = True):
        """Add a fact to the knowledge base"""
        self.facts.append({'predicate': predicate, 'value': value})
    
    def evaluate_compliance(self, fopl_rule: str, compliance_case: Dict,
                           context: Dict) -> ReasoningResult:
        """
        Evaluate if a compliance case satisfies a FOPL rule
        
        Args:
            fopl_rule: The FOPL rule to check
            compliance_case: Dictionary of actual values
            context: Entity mappings
            
        Returns:
            ReasoningResult with outcome and explanation
        """
        # Parse the rule
        parsed_rule = self.parse_fopl(fopl_rule)
        
        # Build fact base from compliance case
        self.clear()
        self._build_facts_from_case(compliance_case, context)
        
        # Evaluate the rule
        proof_trace = []
        satisfied = []
        violated = []
        
        try:
            result, trace = self._evaluate_expression(
                parsed_rule['body'],
                parsed_rule.get('variables', []),
                context,
                compliance_case
            )
            
            proof_trace = trace
            
            if result:
                satisfied.append(fopl_rule)
                explanation = "✅ Compliance check PASSED: All conditions satisfied"
            else:
                violated.append(fopl_rule)
                explanation = "❌ Compliance check FAILED: Conditions not met"
            
            return ReasoningResult(
                outcome=result,
                explanation=explanation,
                proof_trace=proof_trace,
                satisfied_rules=satisfied,
                violated_rules=violated
            )
            
        except Exception as e:
            return ReasoningResult(
                outcome=False,
                explanation=f"⚠️ Evaluation error: {str(e)}",
                proof_trace=[str(e)],
                satisfied_rules=[],
                violated_rules=[fopl_rule]
            )
    
    def _build_facts_from_case(self, compliance_case: Dict, context: Dict):
        """Build facts from compliance case"""
        for entity, facts in compliance_case.items():
            for fact_name, fact_value in facts.items():
                self.add_fact(f"{fact_name}({entity}, {fact_value})")
    
    def _evaluate_expression(self, expr: Dict, variables: List[str],
                            context: Dict, case: Dict) -> Tuple[bool, List[str]]:
        """Recursively evaluate a parsed expression"""
        trace = []
        
        if expr['type'] == 'implication':
            # A -> B: if A is true, then B must be true
            antecedent_result, ant_trace = self._evaluate_expression(
                expr['antecedent'], variables, context, case
            )
            trace.extend([f"  Evaluating antecedent: {t}" for t in ant_trace])
            
            if not antecedent_result:
                # If antecedent is false, implication is vacuously true
                trace.append("  → Antecedent false, implication holds (vacuous truth)")
                return True, trace
            
            # Antecedent is true, check consequent
            consequent_result, cons_trace = self._evaluate_expression(
                expr['consequent'], variables, context, case
            )
            trace.extend([f"  Evaluating consequent: {t}" for t in cons_trace])
            
            trace.append(f"  → Implication result: {consequent_result}")
            return consequent_result, trace
        
        elif expr['type'] == 'conjunction':
            # All operands must be true
            trace.append("Evaluating conjunction (AND):")
            results = []
            for operand in expr['operands']:
                result, op_trace = self._evaluate_expression(
                    operand, variables, context, case
                )
                trace.extend([f"  {t}" for t in op_trace])
                results.append(result)
            
            final_result = all(results)
            trace.append(f"  → Conjunction result: {final_result}")
            return final_result, trace
        
        elif expr['type'] == 'disjunction':
            # At least one operand must be true
            trace.append("Evaluating disjunction (OR):")
            results = []
            for operand in expr['operands']:
                result, op_trace = self._evaluate_expression(
                    operand, variables, context, case
                )
                trace.extend([f"  {t}" for t in op_trace])
                results.append(result)
            
            final_result = any(results)
            trace.append(f"  → Disjunction result: {final_result}")
            return final_result, trace
        
        elif expr['type'] == 'negation':
            # Negate the operand
            result, neg_trace = self._evaluate_expression(
                expr['operand'], variables, context, case
            )
            trace.extend([f"  Negating: {t}" for t in neg_trace])
            final_result = not result
            trace.append(f"  → Negation result: {final_result}")
            return final_result, trace
        
        elif expr['type'] == 'predicate':
            # Evaluate predicate against facts
            result, pred_trace = self._evaluate_predicate(expr, context, case)
            trace.extend(pred_trace)
            return result, trace
        
        else:
            trace.append(f"⚠️ Unknown expression type: {expr['type']}")
            return False, trace
    
    def _evaluate_predicate(self, predicate: Dict, context: Dict,
                           case: Dict) -> Tuple[bool, List[str]]:
        """Evaluate a predicate against the case"""
        trace = []
        pred_name = predicate['name']
        pred_args = predicate['args']
        
        trace.append(f"Checking predicate: {pred_name}")
        
        # Check entity predicates (Tenant, Landlord, etc.)
        if pred_name in context:
            trace.append(f"  ✓ Entity {pred_name} exists in context")
            return True, trace
        
        # Check action predicates with constraints
        for entity, facts in case.items():
            for fact_name, fact_value in facts.items():
                # Check if this fact matches the predicate
                if self._matches_predicate(pred_name, fact_name):
                    # Check constraints
                    if pred_args and len(pred_args) > 1:
                        constraint = pred_args[1]
                        if constraint['op']:
                            result = self._check_constraint(
                                fact_value, constraint['op'], 
                                constraint['value']
                            )
                            trace.append(
                                f"  Constraint: {fact_value} {constraint['op']} "
                                f"{constraint['value']} = {result}"
                            )
                            return result, trace
                    
                    trace.append(f"  ✓ Fact {fact_name} = {fact_value}")
                    return bool(fact_value), trace
        
        trace.append(f"  ✗ No matching facts found")
        return False, trace
    
    def _matches_predicate(self, pred_name: str, fact_name: str) -> bool:
        """Check if a fact name matches a predicate"""
        # Simple matching: PayRent matches PayRentDate, PayRent, etc.
        return (pred_name.lower() in fact_name.lower() or 
                fact_name.lower() in pred_name.lower())
    
    def _check_constraint(self, value: Any, operator: str, 
                         target: str) -> bool:
        """Check if a value satisfies a constraint"""
        try:
            # Convert to numbers if possible
            if isinstance(value, (int, float)):
                target_val = float(target)
            else:
                target_val = target
                value = str(value)
            
            if operator == '<=':
                return value <= target_val
            elif operator == '>=':
                return value >= target_val
            elif operator == '<':
                return value < target_val
            elif operator == '>':
                return value > target_val
            elif operator == '=' or operator == '==':
                return value == target_val
            else:
                return False
        except:
            return False


if __name__ == "__main__":
    # Test the symbolic reasoner
    reasoner = SymbolicReasoner()
    
    # Test case 1: Payment compliance
    fopl_rule = "forall x (Tenant(x) -> PayRent(x, due_date <= 5))"
    context = {"Tenant": "PartyA", "Landlord": "PartyB"}
    
    # Case 1: Compliant (paid on day 4)
    compliance_case_pass = {"PartyA": {"PayRentDate": 4}}
    result = reasoner.evaluate_compliance(fopl_rule, compliance_case_pass, context)
    
    print("Test 1 - Payment on time:")
    print(f"Outcome: {result.outcome}")
    print(f"Explanation: {result.explanation}")
    print("Proof trace:")
    for step in result.proof_trace:
        print(f"  {step}")
    
    print("\n" + "="*60 + "\n")
    
    # Case 2: Non-compliant (paid on day 10)
    compliance_case_fail = {"PartyA": {"PayRentDate": 10}}
    result = reasoner.evaluate_compliance(fopl_rule, compliance_case_fail, context)
    
    print("Test 2 - Payment late:")
    print(f"Outcome: {result.outcome}")
    print(f"Explanation: {result.explanation}")
    print("Proof trace:")
    for step in result.proof_trace:
        print(f"  {step}")