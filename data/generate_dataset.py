import json
import random
from typing import List, Dict

class LegalClauseGenerator:
    def __init__(self):
        self.clause_templates = self._create_clause_templates()
        
    def _create_clause_templates(self) -> List[Dict]:
        """Create diverse legal clause templates"""
        return [
            # Payment obligations
            {
                "text": "The tenant must pay rent by the {day}th of each month.",
                "fopl": "forall x (Tenant(x) -> PayRent(x, due_date <= {day}))",
                "predicates": ["Tenant(x)", "PayRent(x, date)"],
                "type": "payment",
                "variables": {"x": "Tenant"},
                "params": {"day": [1, 5, 10, 15]}
            },
            {
                "text": "The buyer shall pay {amount} USD upon contract signing.",
                "fopl": "forall x (Buyer(x) -> PayAmount(x, {amount}, event=signing))",
                "predicates": ["Buyer(x)", "PayAmount(x, amount, event)"],
                "type": "payment",
                "variables": {"x": "Buyer"},
                "params": {"amount": [1000, 5000, 10000, 50000]}
            },
            # Termination clauses
            {
                "text": "If tenant fails to pay rent, landlord may terminate lease.",
                "fopl": "forall x, y (Tenant(x) & Landlord(y) & ~PayRent(x) -> RightToTerminate(y, x))",
                "predicates": ["Tenant(x)", "Landlord(y)", "PayRent(x)", "RightToTerminate(y, x)"],
                "type": "termination",
                "variables": {"x": "Tenant", "y": "Landlord"},
                "params": {}
            },
            {
                "text": "Either party may terminate with {days} days written notice.",
                "fopl": "forall x (Party(x) -> CanTerminate(x, notice_days >= {days}))",
                "predicates": ["Party(x)", "CanTerminate(x, days)"],
                "type": "termination",
                "variables": {"x": "Party"},
                "params": {"days": [30, 60, 90]}
            },
            # Maintenance obligations
            {
                "text": "The landlord must maintain the property in habitable condition.",
                "fopl": "forall x (Landlord(x) -> MaintainProperty(x, condition=habitable))",
                "predicates": ["Landlord(x)", "MaintainProperty(x, condition)"],
                "type": "maintenance",
                "variables": {"x": "Landlord"},
                "params": {}
            },
            {
                "text": "Tenant shall perform routine maintenance every {months} months.",
                "fopl": "forall x (Tenant(x) -> PerformMaintenance(x, frequency <= {months}))",
                "predicates": ["Tenant(x)", "PerformMaintenance(x, frequency)"],
                "type": "maintenance",
                "variables": {"x": "Tenant"},
                "params": {"months": [3, 6, 12]}
            },
            # Access rights
            {
                "text": "Landlord may enter property with {hours} hours notice for repairs.",
                "fopl": "forall x (Landlord(x) -> CanEnter(x, notice_hours >= {hours}, reason=repairs))",
                "predicates": ["Landlord(x)", "CanEnter(x, notice, reason)"],
                "type": "access",
                "variables": {"x": "Landlord"},
                "params": {"hours": [24, 48, 72]}
            },
            # Liability clauses
            {
                "text": "The seller is liable for defects discovered within {days} days.",
                "fopl": "forall x, y (Seller(x) & Defect(y) & DiscoveredWithin(y, {days}) -> Liable(x, y))",
                "predicates": ["Seller(x)", "Defect(y)", "DiscoveredWithin(y, days)", "Liable(x, y)"],
                "type": "liability",
                "variables": {"x": "Seller", "y": "Defect"},
                "params": {"days": [30, 60, 90, 180]}
            },
            # Confidentiality
            {
                "text": "The employee must not disclose confidential information for {years} years.",
                "fopl": "forall x (Employee(x) -> ~Disclose(x, confidential_info, duration <= {years}))",
                "predicates": ["Employee(x)", "Disclose(x, info, duration)"],
                "type": "confidentiality",
                "variables": {"x": "Employee"},
                "params": {"years": [1, 2, 3, 5]}
            },
            # Insurance requirements
            {
                "text": "The contractor must maintain insurance coverage of at least {amount} USD.",
                "fopl": "forall x (Contractor(x) -> HasInsurance(x, coverage >= {amount}))",
                "predicates": ["Contractor(x)", "HasInsurance(x, coverage)"],
                "type": "insurance",
                "variables": {"x": "Contractor"},
                "params": {"amount": [100000, 500000, 1000000]}
            },
            # Delivery obligations
            {
                "text": "The supplier must deliver goods within {days} business days.",
                "fopl": "forall x (Supplier(x) -> DeliverGoods(x, days <= {days}))",
                "predicates": ["Supplier(x)", "DeliverGoods(x, days)"],
                "type": "delivery",
                "variables": {"x": "Supplier"},
                "params": {"days": [5, 10, 15, 30]}
            },
            # Penalty clauses
            {
                "text": "Late payment incurs a penalty of {percent}% per month.",
                "fopl": "forall x (LatePayment(x) -> Penalty(x, rate={percent}))",
                "predicates": ["LatePayment(x)", "Penalty(x, rate)"],
                "type": "penalty",
                "variables": {"x": "Payment"},
                "params": {"percent": [1, 2, 5, 10]}
            },
            # Warranty clauses
            {
                "text": "The product is warranted for {months} months from purchase date.",
                "fopl": "forall x (Product(x) -> Warranty(x, duration={months}))",
                "predicates": ["Product(x)", "Warranty(x, duration)"],
                "type": "warranty",
                "variables": {"x": "Product"},
                "params": {"months": [6, 12, 24, 36]}
            },
            # Indemnification
            {
                "text": "The contractor indemnifies the client against third-party claims.",
                "fopl": "forall x, y, z (Contractor(x) & Client(y) & ThirdPartyClaim(z) -> Indemnifies(x, y, z))",
                "predicates": ["Contractor(x)", "Client(y)", "ThirdPartyClaim(z)", "Indemnifies(x, y, z)"],
                "type": "indemnification",
                "variables": {"x": "Contractor", "y": "Client", "z": "Claim"},
                "params": {}
            },
            # Non-compete
            {
                "text": "Employee agrees not to compete for {years} years after termination.",
                "fopl": "forall x (Employee(x) & AfterTermination(x) -> ~Compete(x, duration <= {years}))",
                "predicates": ["Employee(x)", "AfterTermination(x)", "Compete(x, duration)"],
                "type": "non_compete",
                "variables": {"x": "Employee"},
                "params": {"years": [1, 2, 3]}
            }
        ]
    
    def generate_clause(self, clause_id: int) -> Dict:
        """Generate a single legal clause with all required fields"""
        template = random.choice(self.clause_templates)
        
        # Fill in parameters if template has params
        clause_text = template["text"]
        fopl_rule = template["fopl"]
        
        param_values = {}
        if template["params"]:
            for param, values in template["params"].items():
                value = random.choice(values)
                param_values[param] = value
                clause_text = clause_text.replace(f"{{{param}}}", str(value))
                fopl_rule = fopl_rule.replace(f"{{{param}}}", str(value))
        
        # Generate context based on variables
        context = {}
        for var, role in template["variables"].items():
            if role in ["Tenant", "Landlord", "Buyer", "Seller"]:
                context[role] = f"Party{chr(65 + len(context))}"
            else:
                context[role] = role + "1"
        
        # Generate compliance case and expected outcome
        compliance_case, expected_outcome = self._generate_compliance_case(
            template, context, param_values
        )
        
        return {
            "id": f"clause_{clause_id:03d}",
            "clause_text": clause_text,
            "context": context,
            "fopl_rule": fopl_rule,
            "predicates_used": template["predicates"],
            "variables": template["variables"],
            "clause_type": template["type"],
            "compliance_case": compliance_case,
            "expected_outcome": expected_outcome
        }
    
    def _generate_compliance_case(self, template: Dict, context: Dict, 
                                   param_values: Dict) -> tuple:
        """Generate a compliance test case based on clause type"""
        compliance_case = {}
        
        if template["type"] == "payment":
            party = list(context.values())[0]
            if "day" in param_values:
                # Random day before or after due date
                actual_day = random.randint(1, 20)
                compliance_case[party] = {"PayRentDate": actual_day}
                expected_outcome = actual_day <= param_values["day"]
            elif "amount" in param_values:
                actual_amount = random.choice([param_values["amount"], param_values["amount"] - 1000])
                compliance_case[party] = {"PaidAmount": actual_amount}
                expected_outcome = actual_amount >= param_values["amount"]
                
        elif template["type"] == "termination":
            if "days" in param_values:
                party = list(context.values())[0]
                notice_given = random.choice([15, 30, 45, 60, 90])
                compliance_case[party] = {"NoticeGiven": notice_given}
                expected_outcome = notice_given >= param_values["days"]
            else:
                party = list(context.values())[0]
                compliance_case[party] = {"PaidRent": random.choice([True, False])}
                expected_outcome = not compliance_case[party]["PaidRent"]
                
        elif template["type"] == "maintenance":
            party = list(context.values())[0]
            if "months" in param_values:
                months_since = random.choice([2, 4, 6, 8, 12])
                compliance_case[party] = {"MonthsSinceMaintenance": months_since}
                expected_outcome = months_since <= param_values["months"]
            else:
                compliance_case[party] = {"PropertyCondition": random.choice(["habitable", "poor"])}
                expected_outcome = compliance_case[party]["PropertyCondition"] == "habitable"
                
        elif template["type"] in ["access", "delivery", "warranty"]:
            party = list(context.values())[0]
            param_key = list(param_values.keys())[0] if param_values else "default"
            if param_key == "hours":
                actual_hours = random.choice([12, 24, 48, 72])
                compliance_case[party] = {"NoticeHours": actual_hours}
                expected_outcome = actual_hours >= param_values["hours"]
            elif param_key == "days":
                actual_days = random.choice([3, 7, 14, 20, 30])
                compliance_case[party] = {"DeliveryDays": actual_days}
                expected_outcome = actual_days <= param_values["days"]
            elif param_key == "months":
                actual_months = random.choice([3, 6, 12, 24, 36])
                compliance_case[party] = {"WarrantyMonths": actual_months}
                expected_outcome = actual_months <= param_values["months"]
                
        else:
            # Default case
            party = list(context.values())[0]
            compliance_case[party] = {"Compliant": random.choice([True, False])}
            expected_outcome = compliance_case[party]["Compliant"]
        
        return compliance_case, expected_outcome
    
    def generate_dataset(self, num_samples: int = 100) -> List[Dict]:
        """Generate the complete dataset"""
        dataset = []
        for i in range(1, num_samples + 1):
            clause = self.generate_clause(i)
            dataset.append(clause)
        return dataset


def main():
    """Generate and save the legal clauses dataset"""
    generator = LegalClauseGenerator()
    dataset = generator.generate_dataset(100)
    
    # Save to JSON file
    with open('legal_clauses.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(dataset)} legal clauses")
    print(f"📁 Saved to: legal_clauses.json")
    
    # Print sample
    print("\n📋 Sample clause:")
    print(json.dumps(dataset[0], indent=2))
    
    # Statistics
    types = {}
    for clause in dataset:
        clause_type = clause["clause_type"]
        types[clause_type] = types.get(clause_type, 0) + 1
    
    print("\n📊 Clause distribution:")
    for clause_type, count in sorted(types.items()):
        print(f"  {clause_type}: {count}")


if __name__ == "__main__":
    main()