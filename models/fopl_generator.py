"""
FOPL (First-Order Predicate Logic) Generator and Utilities
Provides tools for generating, parsing, and validating FOPL expressions
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Quantifier(Enum):
    """FOPL Quantifiers"""
    FORALL = "forall"
    EXISTS = "exists"


class LogicalOperator(Enum):
    """Logical operators"""
    AND = "&"
    OR = "|"
    NOT = "~"
    IMPLIES = "->"
    IFF = "<->"


class ComparisonOperator(Enum):
    """Comparison operators for constraints"""
    LTE = "<="
    GTE = ">="
    LT = "<"
    GT = ">"
    EQ = "="


@dataclass
class Predicate:
    """Represents a logical predicate"""
    name: str
    args: List[str]
    constraints: Dict[str, Tuple[ComparisonOperator, any]] = None
    
    def __str__(self):
        if not self.args:
            return f"{self.name}()"
        
        arg_strs = []
        for i, arg in enumerate(self.args):
            if self.constraints and arg in self.constraints:
                op, value = self.constraints[arg]
                arg_strs.append(f"{arg} {op.value} {value}")
            else:
                arg_strs.append(arg)
        
        return f"{self.name}({', '.join(arg_strs)})"


@dataclass
class FOPLExpression:
    """Represents a complete FOPL expression"""
    quantifier: Quantifier
    variables: List[str]
    body: str
    
    def __str__(self):
        var_str = ', '.join(self.variables)
        return f"{self.quantifier.value} {var_str} ({self.body})"


class FOPLGenerator:
    """
    Generator for FOPL expressions from structured templates
    """
    
    def __init__(self):
        self.predicates = {}
        
    def create_predicate(self, name: str, args: List[str], 
                        constraints: Dict[str, Tuple[str, any]] = None) -> Predicate:
        """
        Create a predicate with optional constraints
        
        Args:
            name: Predicate name (capitalized)
            args: List of argument variables
            constraints: Dict mapping arg to (operator, value)
            
        Example:
            create_predicate("PayRent", ["x", "date"], {"date": ("<=", 5)})
            → PayRent(x, date <= 5)
        """
        constraint_dict = {}
        if constraints:
            for arg, (op, val) in constraints.items():
                constraint_dict[arg] = (ComparisonOperator(op), val)
        
        return Predicate(name, args, constraint_dict)
    
    def create_implication(self, antecedent: str, consequent: str) -> str:
        """
        Create an implication: A -> B
        
        Args:
            antecedent: Left side of implication
            consequent: Right side of implication
            
        Returns:
            String representation
        """
        return f"{antecedent} -> {consequent}"
    
    def create_conjunction(self, *terms: str) -> str:
        """Create conjunction: A & B & C"""
        return " & ".join(terms)
    
    def create_disjunction(self, *terms: str) -> str:
        """Create disjunction: A | B | C"""
        return " | ".join(terms)
    
    def create_negation(self, term: str) -> str:
        """Create negation: ~A"""
        return f"~{term}"
    
    def create_fopl(self, quantifier: str, variables: List[str], 
                   body: str) -> FOPLExpression:
        """
        Create complete FOPL expression
        
        Args:
            quantifier: 'forall' or 'exists'
            variables: List of variable names
            body: Logical expression body
            
        Returns:
            FOPLExpression object
        """
        quant = Quantifier(quantifier)
        return FOPLExpression(quant, variables, body)
    
    def generate_payment_rule(self, entity: str = "x", deadline: int = 5) -> str:
        """Generate a payment obligation rule"""
        pred1 = self.create_predicate("Tenant", [entity])
        pred2 = self.create_predicate("PayRent", [entity, "due_date"], 
                                     {"due_date": ("<=", deadline)})
        
        body = self.create_implication(str(pred1), str(pred2))
        fopl = self.create_fopl("forall", [entity], body)
        
        return str(fopl)
    
    def generate_termination_rule(self, tenant: str = "x", 
                                 landlord: str = "y") -> str:
        """Generate a termination rule"""
        pred1 = self.create_predicate("Tenant", [tenant])
        pred2 = self.create_predicate("Landlord", [landlord])
        pred3 = self.create_predicate("PayRent", [tenant])
        pred4 = self.create_predicate("RightToTerminate", [landlord, tenant])
        
        antecedent = self.create_conjunction(
            str(pred1), 
            str(pred2), 
            self.create_negation(str(pred3))
        )
        
        body = self.create_implication(antecedent, str(pred4))
        fopl = self.create_fopl("forall", [tenant, landlord], body)
        
        return str(fopl)
    
    def generate_constraint_rule(self, entity: str, predicate: str,
                                param: str, operator: str, 
                                value: any) -> str:
        """
        Generate a rule with constraint
        
        Example:
            generate_constraint_rule("x", "Deliver", "days", "<=", 10)
            → forall x (Supplier(x) -> Deliver(x, days <= 10))
        """
        pred1 = self.create_predicate(entity.capitalize(), [entity])
        pred2 = self.create_predicate(predicate, [entity, param], 
                                     {param: (operator, value)})
        
        body = self.create_implication(str(pred1), str(pred2))
        fopl = self.create_fopl("forall", [entity], body)
        
        return str(fopl)


class FOPLParser:
    """
    Parser for FOPL expressions
    Converts string FOPL to structured format
    """
    
    @staticmethod
    def extract_quantifier(fopl_str: str) -> Tuple[Optional[str], str]:
        """Extract quantifier from FOPL string"""
        fopl_str = fopl_str.strip()
        
        if fopl_str.startswith("forall"):
            return "forall", fopl_str[6:].strip()
        elif fopl_str.startswith("exists"):
            return "exists", fopl_str[6:].strip()
        
        return None, fopl_str
    
    @staticmethod
    def extract_variables(fopl_str: str) -> Tuple[List[str], str]:
        """Extract variables from FOPL string"""
        match = re.match(r'([a-z,\s]+)\s*\(', fopl_str)
        
        if match:
            var_str = match.group(1).strip()
            variables = [v.strip() for v in var_str.split(',')]
            remaining = fopl_str[match.end()-1:]
            return variables, remaining
        
        return [], fopl_str
    
    @staticmethod
    def extract_predicates(fopl_str: str) -> List[str]:
        """Extract all predicates from FOPL string"""
        pattern = r'[A-Z][a-zA-Z]*\([^)]*\)'
        return re.findall(pattern, fopl_str)
    
    @staticmethod
    def validate_syntax(fopl_str: str) -> Tuple[bool, str]:
        """
        Validate FOPL syntax
        
        Returns:
            (is_valid, error_message)
        """
        # Check balanced parentheses
        if fopl_str.count('(') != fopl_str.count(')'):
            return False, "Unbalanced parentheses"
        
        # Check for quantifier
        if not any(q in fopl_str for q in ['forall', 'exists']):
            return False, "Missing quantifier (forall/exists)"
        
        # Check for at least one predicate
        if not re.search(r'[A-Z][a-zA-Z]*\(', fopl_str):
            return False, "No predicates found"
        
        # Check for valid operators
        valid_operators = ['->', '&', '|', '~', '<=', '>=', '<', '>', '=']
        # This is a basic check; could be more sophisticated
        
        return True, "Valid FOPL"
    
    @staticmethod
    def parse(fopl_str: str) -> Dict:
        """
        Parse FOPL string into structured format
        
        Returns:
            Dictionary with quantifier, variables, body, predicates
        """
        original = fopl_str
        
        # Extract quantifier
        quantifier, fopl_str = FOPLParser.extract_quantifier(fopl_str)
        
        # Extract variables
        variables, fopl_str = FOPLParser.extract_variables(fopl_str)
        
        # Extract body (remove outer parentheses)
        body = fopl_str.strip()
        if body.startswith('(') and body.endswith(')'):
            body = body[1:-1].strip()
        
        # Extract predicates
        predicates = FOPLParser.extract_predicates(original)
        
        # Validate
        is_valid, message = FOPLParser.validate_syntax(original)
        
        return {
            'original': original,
            'quantifier': quantifier,
            'variables': variables,
            'body': body,
            'predicates': predicates,
            'valid': is_valid,
            'validation_message': message
        }


class FOPLTemplates:
    """
    Pre-defined FOPL templates for common legal clauses
    """
    
    @staticmethod
    def payment_obligation(entity: str = "x", amount: int = None, 
                          deadline: int = None) -> str:
        """Payment obligation template"""
        generator = FOPLGenerator()
        
        if deadline:
            return generator.generate_payment_rule(entity, deadline)
        else:
            pred1 = generator.create_predicate("Payer", [entity])
            if amount:
                pred2 = generator.create_predicate("PayAmount", [entity, "amount"],
                                                  {"amount": (">=", amount)})
            else:
                pred2 = generator.create_predicate("PayAmount", [entity])
            
            body = generator.create_implication(str(pred1), str(pred2))
            return str(generator.create_fopl("forall", [entity], body))
    
    @staticmethod
    def termination_clause(party1: str = "x", party2: str = "y",
                          notice_days: int = None) -> str:
        """Termination clause template"""
        generator = FOPLGenerator()
        
        pred1 = generator.create_predicate("Party", [party1])
        
        if notice_days:
            pred2 = generator.create_predicate("CanTerminate", [party1, "notice"],
                                             {"notice": (">=", notice_days)})
        else:
            pred2 = generator.create_predicate("CanTerminate", [party1])
        
        body = generator.create_implication(str(pred1), str(pred2))
        return str(generator.create_fopl("forall", [party1], body))
    
    @staticmethod
    def liability_clause(liable_party: str = "x", condition: str = "y",
                        duration: int = None) -> str:
        """Liability clause template"""
        generator = FOPLGenerator()
        
        pred1 = generator.create_predicate("Party", [liable_party])
        pred2 = generator.create_predicate("Condition", [condition])
        
        if duration:
            pred3 = generator.create_predicate("Liable", [liable_party, condition, "days"],
                                             {"days": ("<=", duration)})
        else:
            pred3 = generator.create_predicate("Liable", [liable_party, condition])
        
        antecedent = generator.create_conjunction(str(pred1), str(pred2))
        body = generator.create_implication(antecedent, str(pred3))
        
        return str(generator.create_fopl("forall", [liable_party, condition], body))


# Convenience functions
def create_simple_rule(entity: str, predicate: str, 
                      constraint: Optional[Tuple[str, str, any]] = None) -> str:
    """
    Create a simple FOPL rule quickly
    
    Example:
        create_simple_rule("x", "PayRent", ("due_date", "<=", 5))
        → forall x (Entity(x) -> PayRent(x, due_date <= 5))
    """
    generator = FOPLGenerator()
    
    pred1 = generator.create_predicate("Entity", [entity])
    
    if constraint:
        param, op, val = constraint
        pred2 = generator.create_predicate(predicate, [entity, param],
                                         {param: (op, val)})
    else:
        pred2 = generator.create_predicate(predicate, [entity])
    
    body = generator.create_implication(str(pred1), str(pred2))
    fopl = generator.create_fopl("forall", [entity], body)
    
    return str(fopl)


if __name__ == "__main__":
    # Test FOPL Generator
    print("="*60)
    print("Testing FOPL Generator")
    print("="*60 + "\n")
    
    generator = FOPLGenerator()
    
    # Test 1: Payment rule
    print("1. Payment Rule:")
    payment = generator.generate_payment_rule("x", 5)
    print(f"   {payment}\n")
    
    # Test 2: Termination rule
    print("2. Termination Rule:")
    termination = generator.generate_termination_rule("x", "y")
    print(f"   {termination}\n")
    
    # Test 3: Custom constraint rule
    print("3. Custom Constraint Rule:")
    custom = generator.generate_constraint_rule("x", "Deliver", "days", "<=", 10)
    print(f"   {custom}\n")
    
    # Test 4: Templates
    print("4. Using Templates:")
    print(f"   Payment: {FOPLTemplates.payment_obligation('x', deadline=5)}")
    print(f"   Termination: {FOPLTemplates.termination_clause('x', notice_days=30)}")
    print(f"   Liability: {FOPLTemplates.liability_clause('x', 'y', duration=90)}\n")
    
    # Test Parser
    print("="*60)
    print("Testing FOPL Parser")
    print("="*60 + "\n")
    
    parser = FOPLParser()
    
    test_fopl = "forall x (Tenant(x) -> PayRent(x, due_date <= 5))"
    print(f"Parsing: {test_fopl}\n")
    
    parsed = parser.parse(test_fopl)
    print(f"Quantifier: {parsed['quantifier']}")
    print(f"Variables: {parsed['variables']}")
    print(f"Body: {parsed['body']}")
    print(f"Predicates: {parsed['predicates']}")
    print(f"Valid: {parsed['valid']} - {parsed['validation_message']}")