import json
import random
from typing import List, Dict

class LegalClauseGenerator:
    def __init__(self):
        self.clause_templates = self._create_clause_templates()
        
    def _create_clause_templates(self) -> List[Dict]:
        """Create diverse legal clause templates with 50+ examples for rich dataset generation"""
        return [
            # ========== PAYMENT OBLIGATIONS (10 variations) ==========
            {
                "text": "The tenant must pay rent by the {day}th of each month.",
                "fopl": "forall x (Tenant(x) -> PayRent(x, due_date <= {day}))",
                "predicates": ["Tenant(x)", "PayRent(x, date)"],
                "type": "payment",
                "variables": {"x": "Tenant"},
                "params": {"day": [1, 5, 10, 15, 20, 25]}
            },
            {
                "text": "The buyer shall pay {amount} USD upon contract signing.",
                "fopl": "forall x (Buyer(x) -> PayAmount(x, {amount}, event=signing))",
                "predicates": ["Buyer(x)", "PayAmount(x, amount, event)"],
                "type": "payment",
                "variables": {"x": "Buyer"},
                "params": {"amount": [1000, 5000, 10000, 25000, 50000, 100000]}
            },
            {
                "text": "Monthly installment of {amount} USD must be paid within {days} days.",
                "fopl": "forall x (Debtor(x) -> PayInstallment(x, amount={amount}, deadline <= {days}))",
                "predicates": ["Debtor(x)", "PayInstallment(x, amount, deadline)"],
                "type": "payment",
                "variables": {"x": "Debtor"},
                "params": {"amount": [500, 1000, 2000, 5000], "days": [7, 10, 15, 30]}
            },
            {
                "text": "The contractor shall invoice within {days} days of work completion.",
                "fopl": "forall x (Contractor(x) -> SubmitInvoice(x, deadline <= {days}))",
                "predicates": ["Contractor(x)", "SubmitInvoice(x, deadline)"],
                "type": "payment",
                "variables": {"x": "Contractor"},
                "params": {"days": [7, 14, 21, 30, 45]}
            },
            {
                "text": "Payment shall be made in {installments} equal installments.",
                "fopl": "forall x (Payer(x) -> PayInInstallments(x, count={installments}))",
                "predicates": ["Payer(x)", "PayInInstallments(x, count)"],
                "type": "payment",
                "variables": {"x": "Payer"},
                "params": {"installments": [2, 3, 4, 6, 12]}
            },
            {
                "text": "Advance payment of {percent}% is required before service delivery.",
                "fopl": "forall x (Client(x) -> PayAdvance(x, percentage={percent}))",
                "predicates": ["Client(x)", "PayAdvance(x, percentage)"],
                "type": "payment",
                "variables": {"x": "Client"},
                "params": {"percent": [10, 20, 25, 30, 50]}
            },
            {
                "text": "Security deposit of {amount} USD must be paid upfront.",
                "fopl": "forall x (Tenant(x) -> PayDeposit(x, amount={amount}))",
                "predicates": ["Tenant(x)", "PayDeposit(x, amount)"],
                "type": "payment",
                "variables": {"x": "Tenant"},
                "params": {"amount": [500, 1000, 2000, 3000, 5000]}
            },
            {
                "text": "Late payment fee of {amount} USD applies after {days} days.",
                "fopl": "forall x (Debtor(x) & LatePayment(x, days > {days}) -> PayFee(x, amount={amount}))",
                "predicates": ["Debtor(x)", "LatePayment(x, days)", "PayFee(x, amount)"],
                "type": "payment",
                "variables": {"x": "Debtor"},
                "params": {"amount": [25, 50, 100, 200], "days": [5, 10, 15, 30]}
            },
            {
                "text": "The purchaser agrees to pay {percent}% down payment.",
                "fopl": "forall x (Purchaser(x) -> PayDownPayment(x, percentage={percent}))",
                "predicates": ["Purchaser(x)", "PayDownPayment(x, percentage)"],
                "type": "payment",
                "variables": {"x": "Purchaser"},
                "params": {"percent": [10, 15, 20, 25, 30, 40, 50]}
            },
            {
                "text": "Annual subscription fee of {amount} USD is due each year.",
                "fopl": "forall x (Subscriber(x) -> PayAnnualFee(x, amount={amount}))",
                "predicates": ["Subscriber(x)", "PayAnnualFee(x, amount)"],
                "type": "payment",
                "variables": {"x": "Subscriber"},
                "params": {"amount": [100, 250, 500, 1000, 2000, 5000]}
            },
            
            # ========== TERMINATION CLAUSES (8 variations) ==========
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
                "params": {"days": [15, 30, 45, 60, 90, 120]}
            },
            {
                "text": "Contract automatically terminates after {years} years unless renewed.",
                "fopl": "forall x (Contract(x) & Duration(x, years >= {years}) -> AutoTerminate(x))",
                "predicates": ["Contract(x)", "Duration(x, years)", "AutoTerminate(x)"],
                "type": "termination",
                "variables": {"x": "Contract"},
                "params": {"years": [1, 2, 3, 5, 10]}
            },
            {
                "text": "Material breach permits immediate termination without notice.",
                "fopl": "forall x, y (Party(x) & MaterialBreach(y) -> CanTerminateImmediately(x))",
                "predicates": ["Party(x)", "MaterialBreach(y)", "CanTerminateImmediately(x)"],
                "type": "termination",
                "variables": {"x": "Party", "y": "Breach"},
                "params": {}
            },
            {
                "text": "Termination for convenience requires {months} months advance notice.",
                "fopl": "forall x (Party(x) & TerminateForConvenience(x) -> GiveNotice(x, months >= {months}))",
                "predicates": ["Party(x)", "TerminateForConvenience(x)", "GiveNotice(x, months)"],
                "type": "termination",
                "variables": {"x": "Party"},
                "params": {"months": [1, 2, 3, 6]}
            },
            {
                "text": "Failure to cure breach within {days} days allows termination.",
                "fopl": "forall x (Party(x) & ~CureBreach(x, days <= {days}) -> AllowTermination(x))",
                "predicates": ["Party(x)", "CureBreach(x, days)", "AllowTermination(x)"],
                "type": "termination",
                "variables": {"x": "Party"},
                "params": {"days": [15, 30, 45, 60]}
            },
            {
                "text": "Employee may resign with {weeks} weeks notice period.",
                "fopl": "forall x (Employee(x) -> CanResign(x, notice_weeks >= {weeks}))",
                "predicates": ["Employee(x)", "CanResign(x, notice_weeks)"],
                "type": "termination",
                "variables": {"x": "Employee"},
                "params": {"weeks": [1, 2, 3, 4, 8]}
            },
            {
                "text": "Mutual consent required for early termination before {months} months.",
                "fopl": "forall x, y (Party(x) & Party(y) & EarlyTermination(months < {months}) -> RequireMutualConsent(x, y))",
                "predicates": ["Party(x)", "Party(y)", "EarlyTermination(months)", "RequireMutualConsent(x, y)"],
                "type": "termination",
                "variables": {"x": "PartyA", "y": "PartyB"},
                "params": {"months": [3, 6, 12, 24]}
            },
            
            # ========== MAINTENANCE OBLIGATIONS (6 variations) ==========
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
                "params": {"months": [1, 3, 6, 12]}
            },
            {
                "text": "Equipment must be serviced at least {times} times per year.",
                "fopl": "forall x (Equipment(x) -> ServiceRequired(x, frequency >= {times}))",
                "predicates": ["Equipment(x)", "ServiceRequired(x, frequency)"],
                "type": "maintenance",
                "variables": {"x": "Equipment"},
                "params": {"times": [1, 2, 4, 6, 12]}
            },
            {
                "text": "Owner responsible for all repairs exceeding {amount} USD.",
                "fopl": "forall x (Owner(x) & RepairCost(x, cost > {amount}) -> ResponsibleForRepair(x))",
                "predicates": ["Owner(x)", "RepairCost(x, cost)", "ResponsibleForRepair(x)"],
                "type": "maintenance",
                "variables": {"x": "Owner"},
                "params": {"amount": [100, 250, 500, 1000, 2000]}
            },
            {
                "text": "Preventive maintenance must be completed within {days} days of notice.",
                "fopl": "forall x (ServiceProvider(x) -> CompleteMaintenance(x, deadline <= {days}))",
                "predicates": ["ServiceProvider(x)", "CompleteMaintenance(x, deadline)"],
                "type": "maintenance",
                "variables": {"x": "ServiceProvider"},
                "params": {"days": [5, 7, 10, 14, 30]}
            },
            {
                "text": "Annual inspection and maintenance is mandatory every {months} months.",
                "fopl": "forall x (Facility(x) -> AnnualInspection(x, interval <= {months}))",
                "predicates": ["Facility(x)", "AnnualInspection(x, interval)"],
                "type": "maintenance",
                "variables": {"x": "Facility"},
                "params": {"months": [6, 12, 18, 24]}
            },
            
            # ========== ACCESS RIGHTS (5 variations) ==========
            {
                "text": "Landlord may enter property with {hours} hours notice for repairs.",
                "fopl": "forall x (Landlord(x) -> CanEnter(x, notice_hours >= {hours}, reason=repairs))",
                "predicates": ["Landlord(x)", "CanEnter(x, notice, reason)"],
                "type": "access",
                "variables": {"x": "Landlord"},
                "params": {"hours": [12, 24, 48, 72]}
            },
            {
                "text": "Authorized personnel may access premises during business hours with ID.",
                "fopl": "forall x (AuthorizedPersonnel(x) & BusinessHours() & HasID(x) -> CanAccess(x))",
                "predicates": ["AuthorizedPersonnel(x)", "BusinessHours()", "HasID(x)", "CanAccess(x)"],
                "type": "access",
                "variables": {"x": "Personnel"},
                "params": {}
            },
            {
                "text": "Emergency access permitted without prior notice at any time.",
                "fopl": "forall x (Emergency(x) -> AccessWithoutNotice(x))",
                "predicates": ["Emergency(x)", "AccessWithoutNotice(x)"],
                "type": "access",
                "variables": {"x": "Emergency"},
                "params": {}
            },
            {
                "text": "Tenant must provide access for inspection with {days} days notice.",
                "fopl": "forall x (Tenant(x) -> ProvideAccess(x, notice_days >= {days}))",
                "predicates": ["Tenant(x)", "ProvideAccess(x, notice_days)"],
                "type": "access",
                "variables": {"x": "Tenant"},
                "params": {"days": [1, 3, 5, 7, 14]}
            },
            {
                "text": "Restricted areas require {level} security clearance for entry.",
                "fopl": "forall x (RestrictedArea(x) -> RequireClearance(x, level={level}))",
                "predicates": ["RestrictedArea(x)", "RequireClearance(x, level)"],
                "type": "access",
                "variables": {"x": "Area"},
                "params": {"level": [1, 2, 3, 4, 5]}
            },
            
            # ========== LIABILITY CLAUSES (7 variations) ==========
            {
                "text": "The seller is liable for defects discovered within {days} days.",
                "fopl": "forall x, y (Seller(x) & Defect(y) & DiscoveredWithin(y, {days}) -> Liable(x, y))",
                "predicates": ["Seller(x)", "Defect(y)", "DiscoveredWithin(y, days)", "Liable(x, y)"],
                "type": "liability",
                "variables": {"x": "Seller", "y": "Defect"},
                "params": {"days": [30, 60, 90, 180, 365]}
            },
            {
                "text": "Liability is limited to {amount} USD for indirect damages.",
                "fopl": "forall x (Party(x) & IndirectDamages(x) -> LiabilityLimit(x, max={amount}))",
                "predicates": ["Party(x)", "IndirectDamages(x)", "LiabilityLimit(x, max)"],
                "type": "liability",
                "variables": {"x": "Party"},
                "params": {"amount": [1000, 5000, 10000, 50000, 100000]}
            },
            {
                "text": "Provider liable for data breaches resulting in losses.",
                "fopl": "forall x (Provider(x) & DataBreach(x) -> LiableForLosses(x))",
                "predicates": ["Provider(x)", "DataBreach(x)", "LiableForLosses(x)"],
                "type": "liability",
                "variables": {"x": "Provider"},
                "params": {}
            },
            {
                "text": "No liability for delays caused by force majeure events.",
                "fopl": "forall x (Party(x) & ForceMajeure(x) -> ~LiableForDelay(x))",
                "predicates": ["Party(x)", "ForceMajeure(x)", "LiableForDelay(x)"],
                "type": "liability",
                "variables": {"x": "Party"},
                "params": {}
            },
            {
                "text": "Manufacturer liable for product defects for {months} months.",
                "fopl": "forall x (Manufacturer(x) & ProductDefect(x) -> Liable(x, duration <= {months}))",
                "predicates": ["Manufacturer(x)", "ProductDefect(x)", "Liable(x, duration)"],
                "type": "liability",
                "variables": {"x": "Manufacturer"},
                "params": {"months": [6, 12, 24, 36, 60]}
            },
            {
                "text": "Aggregate liability shall not exceed {percent}% of contract value.",
                "fopl": "forall x (Liability(x) -> NotExceed(x, percentage <= {percent}))",
                "predicates": ["Liability(x)", "NotExceed(x, percentage)"],
                "type": "liability",
                "variables": {"x": "Liability"},
                "params": {"percent": [50, 75, 100, 150, 200]}
            },
            {
                "text": "Claims must be filed within {days} days of discovering damage.",
                "fopl": "forall x (Claimant(x) -> FileClaim(x, deadline <= {days}))",
                "predicates": ["Claimant(x)", "FileClaim(x, deadline)"],
                "type": "liability",
                "variables": {"x": "Claimant"},
                "params": {"days": [30, 60, 90, 120, 180]}
            },
            
            # ========== CONFIDENTIALITY (6 variations) ==========
            {
                "text": "The employee must not disclose confidential information for {years} years.",
                "fopl": "forall x (Employee(x) -> ~Disclose(x, confidential_info, duration <= {years}))",
                "predicates": ["Employee(x)", "Disclose(x, info, duration)"],
                "type": "confidentiality",
                "variables": {"x": "Employee"},
                "params": {"years": [1, 2, 3, 5, 10]}
            },
            {
                "text": "All proprietary information remains confidential indefinitely.",
                "fopl": "forall x (ProprietaryInfo(x) -> RemainConfidential(x, duration=indefinite))",
                "predicates": ["ProprietaryInfo(x)", "RemainConfidential(x, duration)"],
                "type": "confidentiality",
                "variables": {"x": "Information"},
                "params": {}
            },
            {
                "text": "Non-disclosure obligation survives termination for {years} years.",
                "fopl": "forall x (Party(x) & AfterTermination(x) -> MaintainNDA(x, years <= {years}))",
                "predicates": ["Party(x)", "AfterTermination(x)", "MaintainNDA(x, years)"],
                "type": "confidentiality",
                "variables": {"x": "Party"},
                "params": {"years": [2, 3, 5, 7, 10]}
            },
            {
                "text": "Confidential data must be returned within {days} days of termination.",
                "fopl": "forall x (Party(x) & Termination(x) -> ReturnData(x, deadline <= {days}))",
                "predicates": ["Party(x)", "Termination(x)", "ReturnData(x, deadline)"],
                "type": "confidentiality",
                "variables": {"x": "Party"},
                "params": {"days": [7, 14, 30, 60]}
            },
            {
                "text": "Breach of confidentiality results in damages of {amount} USD.",
                "fopl": "forall x (Party(x) & BreachConfidentiality(x) -> PayDamages(x, amount={amount}))",
                "predicates": ["Party(x)", "BreachConfidentiality(x)", "PayDamages(x, amount)"],
                "type": "confidentiality",
                "variables": {"x": "Party"},
                "params": {"amount": [10000, 25000, 50000, 100000, 500000]}
            },
            {
                "text": "Permitted disclosure only with {days} days prior written consent.",
                "fopl": "forall x (Party(x) & WantsDisclose(x) -> GetConsent(x, notice_days >= {days}))",
                "predicates": ["Party(x)", "WantsDisclose(x)", "GetConsent(x, notice_days)"],
                "type": "confidentiality",
                "variables": {"x": "Party"},
                "params": {"days": [5, 7, 10, 15, 30]}
            },
            
            # ========== INSURANCE REQUIREMENTS (5 variations) ==========
            {
                "text": "The contractor must maintain insurance coverage of at least {amount} USD.",
                "fopl": "forall x (Contractor(x) -> HasInsurance(x, coverage >= {amount}))",
                "predicates": ["Contractor(x)", "HasInsurance(x, coverage)"],
                "type": "insurance",
                "variables": {"x": "Contractor"},
                "params": {"amount": [100000, 250000, 500000, 1000000, 2000000]}
            },
            {
                "text": "Proof of insurance must be provided within {days} days of request.",
                "fopl": "forall x (Insured(x) -> ProvideProof(x, deadline <= {days}))",
                "predicates": ["Insured(x)", "ProvideProof(x, deadline)"],
                "type": "insurance",
                "variables": {"x": "Insured"},
                "params": {"days": [5, 7, 10, 15, 30]}
            },
            {
                "text": "Liability insurance with {amount} USD per occurrence is mandatory.",
                "fopl": "forall x (Party(x) -> LiabilityInsurance(x, per_occurrence >= {amount}))",
                "predicates": ["Party(x)", "LiabilityInsurance(x, per_occurrence)"],
                "type": "insurance",
                "variables": {"x": "Party"},
                "params": {"amount": [500000, 1000000, 2000000, 5000000]}
            },
            {
                "text": "Insurance policy must name client as additional insured party.",
                "fopl": "forall x, y (Contractor(x) & Client(y) -> NameAsInsured(x, y))",
                "predicates": ["Contractor(x)", "Client(y)", "NameAsInsured(x, y)"],
                "type": "insurance",
                "variables": {"x": "Contractor", "y": "Client"},
                "params": {}
            },
            {
                "text": "Workers compensation insurance required for teams exceeding {workers} workers.",
                "fopl": "forall x (Employer(x) & Workers(x, count > {workers}) -> WorkersCompRequired(x))",
                "predicates": ["Employer(x)", "Workers(x, count)", "WorkersCompRequired(x)"],
                "type": "insurance",
                "variables": {"x": "Employer"},
                "params": {"workers": [3, 5, 10, 15, 20]}
            },
            
            # ========== DELIVERY OBLIGATIONS (7 variations) ==========
            {
                "text": "The supplier must deliver goods within {days} business days.",
                "fopl": "forall x (Supplier(x) -> DeliverGoods(x, days <= {days}))",
                "predicates": ["Supplier(x)", "DeliverGoods(x, days)"],
                "type": "delivery",
                "variables": {"x": "Supplier"},
                "params": {"days": [3, 5, 7, 10, 15, 21, 30, 45]}
            },
            {
                "text": "Delivery must include tracking information and signature confirmation.",
                "fopl": "forall x (Delivery(x) -> IncludeTracking(x) & RequireSignature(x))",
                "predicates": ["Delivery(x)", "IncludeTracking(x)", "RequireSignature(x)"],
                "type": "delivery",
                "variables": {"x": "Delivery"},
                "params": {}
            },
            {
                "text": "Partial shipments allowed with {days} days between installments.",
                "fopl": "forall x (Shipment(x) -> PartialAllowed(x, interval <= {days}))",
                "predicates": ["Shipment(x)", "PartialAllowed(x, interval)"],
                "type": "delivery",
                "variables": {"x": "Shipment"},
                "params": {"days": [7, 14, 21, 30]}
            },
            {
                "text": "Goods must be delivered to specified location by {hour} {period}.",
                "fopl": "forall x (Goods(x) -> DeliverByTime(x, hour={hour}, period={period}))",
                "predicates": ["Goods(x)", "DeliverByTime(x, hour, period)"],
                "type": "delivery",
                "variables": {"x": "Goods"},
                "params": {"hour": [9, 12, 5, 6], "period": ["AM", "PM"]}
            },
            {
                "text": "Expedited shipping available for additional {amount} USD fee.",
                "fopl": "forall x (Customer(x) & ExpeditedShipping(x) -> PayFee(x, amount={amount}))",
                "predicates": ["Customer(x)", "ExpeditedShipping(x)", "PayFee(x, amount)"],
                "type": "delivery",
                "variables": {"x": "Customer"},
                "params": {"amount": [25, 50, 75, 100, 200]}
            },
            {
                "text": "Buyer must inspect goods within {days} days of delivery.",
                "fopl": "forall x (Buyer(x) -> InspectGoods(x, deadline <= {days}))",
                "predicates": ["Buyer(x)", "InspectGoods(x, deadline)"],
                "type": "delivery",
                "variables": {"x": "Buyer"},
                "params": {"days": [3, 5, 7, 10, 14]}
            },
            {
                "text": "Free shipping for orders exceeding {amount} USD value.",
                "fopl": "forall x (Order(x) & Value(x, amount > {amount}) -> FreeShipping(x))",
                "predicates": ["Order(x)", "Value(x, amount)", "FreeShipping(x)"],
                "type": "delivery",
                "variables": {"x": "Order"},
                "params": {"amount": [50, 100, 250, 500, 1000]}
            },
            
            # ========== PENALTY CLAUSES (6 variations) ==========
            {
                "text": "Late payment incurs a penalty of {percent}% per month.",
                "fopl": "forall x (LatePayment(x) -> Penalty(x, rate={percent}))",
                "predicates": ["LatePayment(x)", "Penalty(x, rate)"],
                "type": "penalty",
                "variables": {"x": "Payment"},
                "params": {"percent": [1, 1.5, 2, 3, 5, 10]}
            },
            {
                "text": "Delay penalty of {amount} USD per day after deadline.",
                "fopl": "forall x (Contractor(x) & DelayedCompletion(x) -> PayPenalty(x, per_day={amount}))",
                "predicates": ["Contractor(x)", "DelayedCompletion(x)", "PayPenalty(x, per_day)"],
                "type": "penalty",
                "variables": {"x": "Contractor"},
                "params": {"amount": [100, 250, 500, 1000, 2500]}
            },
            {
                "text": "Cancellation within {hours} hours incurs {percent}% cancellation fee.",
                "fopl": "forall x (Booking(x) & Cancel(x, hours < {hours}) -> CancellationFee(x, rate={percent}))",
                "predicates": ["Booking(x)", "Cancel(x, hours)", "CancellationFee(x, rate)"],
                "type": "penalty",
                "variables": {"x": "Booking"},
                "params": {"hours": [24, 48, 72], "percent": [10, 25, 50, 100]}
            },
            {
                "text": "Non-performance results in liquidated damages of {amount} USD.",
                "fopl": "forall x (Party(x) & NonPerformance(x) -> LiquidatedDamages(x, amount={amount}))",
                "predicates": ["Party(x)", "NonPerformance(x)", "LiquidatedDamages(x, amount)"],
                "type": "penalty",
                "variables": {"x": "Party"},
                "params": {"amount": [5000, 10000, 25000, 50000, 100000]}
            },
            {
                "text": "Maximum penalty shall not exceed {percent}% of total contract value.",
                "fopl": "forall x (Penalty(x) -> MaxLimit(x, percentage <= {percent}))",
                "predicates": ["Penalty(x)", "MaxLimit(x, percentage)"],
                "type": "penalty",
                "variables": {"x": "Penalty"},
                "params": {"percent": [5, 10, 15, 20, 25]}
            },
            {
                "text": "Early withdrawal penalty of {amount} USD applies before {months} months.",
                "fopl": "forall x (Account(x) & Withdraw(x, months < {months}) -> Penalty(x, amount={amount}))",
                "predicates": ["Account(x)", "Withdraw(x, months)", "Penalty(x, amount)"],
                "type": "penalty",
                "variables": {"x": "Account"},
                "params": {"amount": [50, 100, 250, 500], "months": [3, 6, 12, 24]}
            },
            
            # ========== WARRANTY CLAUSES (6 variations) ==========
            {
                "text": "The product is warranted for {months} months from purchase date.",
                "fopl": "forall x (Product(x) -> Warranty(x, duration={months}))",
                "predicates": ["Product(x)", "Warranty(x, duration)"],
                "type": "warranty",
                "variables": {"x": "Product"},
                "params": {"months": [3, 6, 12, 18, 24, 36, 60]}
            },
            {
                "text": "Limited warranty covers manufacturing defects only.",
                "fopl": "forall x (Product(x) & ManufacturingDefect(x) -> WarrantyCovered(x))",
                "predicates": ["Product(x)", "ManufacturingDefect(x)", "WarrantyCovered(x)"],
                "type": "warranty",
                "variables": {"x": "Product"},
                "params": {}
            },
            {
                "text": "Extended warranty available for {amount} USD additional cost.",
                "fopl": "forall x (Customer(x) & WantsExtendedWarranty(x) -> PayFee(x, amount={amount}))",
                "predicates": ["Customer(x)", "WantsExtendedWarranty(x)", "PayFee(x, amount)"],
                "type": "warranty",
                "variables": {"x": "Customer"},
                "params": {"amount": [50, 100, 200, 500, 1000]}
            },
            {
                "text": "Warranty void if product tampered or misused.",
                "fopl": "forall x (Product(x) & (Tampered(x) | Misused(x)) -> VoidWarranty(x))",
                "predicates": ["Product(x)", "Tampered(x)", "Misused(x)", "VoidWarranty(x)"],
                "type": "warranty",
                "variables": {"x": "Product"},
                "params": {}
            },
            {
                "text": "Replacement or repair within {days} days under warranty.",
                "fopl": "forall x (Product(x) & UnderWarranty(x) -> RepairOrReplace(x, deadline <= {days}))",
                "predicates": ["Product(x)", "UnderWarranty(x)", "RepairOrReplace(x, deadline)"],
                "type": "warranty",
                "variables": {"x": "Product"},
                "params": {"days": [7, 14, 21, 30, 45]}
            },
            {
                "text": "Lifetime warranty on parts, {years} years on labor.",
                "fopl": "forall x (Product(x) -> LifetimePartsWarranty(x) & LaborWarranty(x, years={years}))",
                "predicates": ["Product(x)", "LifetimePartsWarranty(x)", "LaborWarranty(x, years)"],
                "type": "warranty",
                "variables": {"x": "Product"},
                "params": {"years": [1, 2, 3, 5]}
            },
            
            # ========== INDEMNIFICATION (5 variations) ==========
            {
                "text": "The contractor indemnifies the client against third-party claims.",
                "fopl": "forall x, y, z (Contractor(x) & Client(y) & ThirdPartyClaim(z) -> Indemnifies(x, y, z))",
                "predicates": ["Contractor(x)", "Client(y)", "ThirdPartyClaim(z)", "Indemnifies(x, y, z)"],
                "type": "indemnification",
                "variables": {"x": "Contractor", "y": "Client", "z": "Claim"},
                "params": {}
            },
            {
                "text": "Indemnity obligations survive termination for {years} years.",
                "fopl": "forall x (Party(x) & AfterTermination(x) -> IndemnityObligationContinues(x, years <= {years}))",
                "predicates": ["Party(x)", "AfterTermination(x)", "IndemnityObligationContinues(x, years)"],
                "type": "indemnification",
                "variables": {"x": "Party"},
                "params": {"years": [2, 3, 5, 7, 10]}
            },
            {
                "text": "Each party indemnifies the other for own negligent acts.",
                "fopl": "forall x, y (Party(x) & Party(y) & Negligence(x) -> Indemnify(x, y))",
                "predicates": ["Party(x)", "Party(y)", "Negligence(x)", "Indemnify(x, y)"],
                "type": "indemnification",
                "variables": {"x": "PartyA", "y": "PartyB"},
                "params": {}
            },
            {
                "text": "Indemnification includes legal fees up to {amount} USD.",
                "fopl": "forall x (Indemnity(x) -> CoversLegalFees(x, max={amount}))",
                "predicates": ["Indemnity(x)", "CoversLegalFees(x, max)"],
                "type": "indemnification",
                "variables": {"x": "Indemnity"},
                "params": {"amount": [10000, 25000, 50000, 100000, 250000]}
            },
            {
                "text": "Notice of claim required within {days} days for indemnification.",
                "fopl": "forall x (Indemnitee(x) & Claim(x) -> GiveNotice(x, deadline <= {days}))",
                "predicates": ["Indemnitee(x)", "Claim(x)", "GiveNotice(x, deadline)"],
                "type": "indemnification",
                "variables": {"x": "Indemnitee"},
                "params": {"days": [10, 15, 30, 60, 90]}
            },
            
            # ========== NON-COMPETE (5 variations) ==========
            {
                "text": "Employee agrees not to compete for {years} years after termination.",
                "fopl": "forall x (Employee(x) & AfterTermination(x) -> ~Compete(x, duration <= {years}))",
                "predicates": ["Employee(x)", "AfterTermination(x)", "Compete(x, duration)"],
                "type": "non_compete",
                "variables": {"x": "Employee"},
                "params": {"years": [1, 2, 3, 5]}
            },
            {
                "text": "Non-compete restricted to {miles} mile radius from office.",
                "fopl": "forall x (Employee(x) & AfterTermination(x) -> NonCompeteRadius(x, miles <= {miles}))",
                "predicates": ["Employee(x)", "AfterTermination(x)", "NonCompeteRadius(x, miles)"],
                "type": "non_compete",
                "variables": {"x": "Employee"},
                "params": {"miles": [10, 25, 50, 100, 200]}
            },
            {
                "text": "Violation of non-compete results in {amount} USD penalty.",
                "fopl": "forall x (Employee(x) & ViolateNonCompete(x) -> PayPenalty(x, amount={amount}))",
                "predicates": ["Employee(x)", "ViolateNonCompete(x)", "PayPenalty(x, amount)"],
                "type": "non_compete",
                "variables": {"x": "Employee"},
                "params": {"amount": [10000, 25000, 50000, 100000]}
            },
            {
                "text": "Non-compete applies only to direct competitors in same industry.",
                "fopl": "forall x (Employee(x) & DirectCompetitor(x) & SameIndustry(x) -> NonCompeteApplies(x))",
                "predicates": ["Employee(x)", "DirectCompetitor(x)", "SameIndustry(x)", "NonCompeteApplies(x)"],
                "type": "non_compete",
                "variables": {"x": "Employee"},
                "params": {}
            },
            {
                "text": "Severance payment of {months} months salary waives non-compete.",
                "fopl": "forall x (Employee(x) & SeverancePaid(x, months >= {months}) -> NonCompeteWaived(x))",
                "predicates": ["Employee(x)", "SeverancePaid(x, months)", "NonCompeteWaived(x)"],
                "type": "non_compete",
                "variables": {"x": "Employee"},
                "params": {"months": [3, 6, 9, 12]}
            },
            
            # ========== INTELLECTUAL PROPERTY (6 variations) ==========
            {
                "text": "All work product belongs to employer as work-for-hire.",
                "fopl": "forall x, y (Employee(x) & Employer(y) & WorkProduct(x) -> BelongsTo(y, x))",
                "predicates": ["Employee(x)", "Employer(y)", "WorkProduct(x)", "BelongsTo(y, x)"],
                "type": "intellectual_property",
                "variables": {"x": "Employee", "y": "Employer"},
                "params": {}
            },
            {
                "text": "Inventor retains {percent}% ownership of patent rights.",
                "fopl": "forall x (Inventor(x) -> RetainOwnership(x, percentage={percent}))",
                "predicates": ["Inventor(x)", "RetainOwnership(x, percentage)"],
                "type": "intellectual_property",
                "variables": {"x": "Inventor"},
                "params": {"percent": [10, 20, 25, 33, 50, 75]}
            },
            {
                "text": "Copyright assignment effective upon payment of {amount} USD.",
                "fopl": "forall x (Creator(x) & PaymentReceived(x, amount >= {amount}) -> AssignCopyright(x))",
                "predicates": ["Creator(x)", "PaymentReceived(x, amount)", "AssignCopyright(x)"],
                "type": "intellectual_property",
                "variables": {"x": "Creator"},
                "params": {"amount": [1000, 5000, 10000, 25000, 50000]}
            },
            {
                "text": "Licensee may use trademark for {years} years only.",
                "fopl": "forall x (Licensee(x) -> UseTrademark(x, duration <= {years}))",
                "predicates": ["Licensee(x)", "UseTrademark(x, duration)"],
                "type": "intellectual_property",
                "variables": {"x": "Licensee"},
                "params": {"years": [1, 2, 3, 5, 10, 20]}
            },
            {
                "text": "Patent infringement results in royalty of {percent}% of revenue.",
                "fopl": "forall x (Infringer(x) -> PayRoyalty(x, rate={percent}))",
                "predicates": ["Infringer(x)", "PayRoyalty(x, rate)"],
                "type": "intellectual_property",
                "variables": {"x": "Infringer"},
                "params": {"percent": [5, 10, 15, 20, 25, 30]}
            },
            {
                "text": "Joint invention ownership split {percent}/{remainder} between parties.",
                "fopl": "forall x, y (Inventor(x) & Inventor(y) -> OwnershipSplit(x, {percent}, y, {remainder}))",
                "predicates": ["Inventor(x)", "Inventor(y)", "OwnershipSplit(x, percent, y, remainder)"],
                "type": "intellectual_property",
                "variables": {"x": "InventorA", "y": "InventorB"},
                "params": {"percent": [25, 33, 40, 50, 60, 67, 75], "remainder": [75, 67, 60, 50, 40, 33, 25]}
            },
            
            # ========== DISPUTE RESOLUTION (7 variations) ==========
            {
                "text": "Disputes shall be resolved through binding arbitration.",
                "fopl": "forall x (Dispute(x) -> ResolveByArbitration(x))",
                "predicates": ["Dispute(x)", "ResolveByArbitration(x)"],
                "type": "dispute_resolution",
                "variables": {"x": "Dispute"},
                "params": {}
            },
            {
                "text": "Mediation required before initiating legal proceedings.",
                "fopl": "forall x (Party(x) & WantsLitigation(x) -> MustMediateFirst(x))",
                "predicates": ["Party(x)", "WantsLitigation(x)", "MustMediateFirst(x)"],
                "type": "dispute_resolution",
                "variables": {"x": "Party"},
                "params": {}
            },
            {
                "text": "Arbitration must commence within {days} days of dispute notice.",
                "fopl": "forall x (Dispute(x) -> CommenceArbitration(x, deadline <= {days}))",
                "predicates": ["Dispute(x)", "CommenceArbitration(x, deadline)"],
                "type": "dispute_resolution",
                "variables": {"x": "Dispute"},
                "params": {"days": [30, 45, 60, 90, 120]}
            },
            {
                "text": "Governing law shall be the laws of {jurisdiction}.",
                "fopl": "forall x (Contract(x) -> GoverningLaw(x, jurisdiction={jurisdiction}))",
                "predicates": ["Contract(x)", "GoverningLaw(x, jurisdiction)"],
                "type": "dispute_resolution",
                "variables": {"x": "Contract"},
                "params": {"jurisdiction": ["NewYork", "California", "Delaware", "Texas", "Florida"]}
            },
            {
                "text": "Each party bears own legal costs unless prevailing party.",
                "fopl": "forall x, y (Party(x) & Party(y) & PrevailingParty(y) -> PayCosts(x, y))",
                "predicates": ["Party(x)", "Party(y)", "PrevailingParty(y)", "PayCosts(x, y)"],
                "type": "dispute_resolution",
                "variables": {"x": "PartyA", "y": "PartyB"},
                "params": {}
            },
            {
                "text": "Arbitration conducted by panel of {arbitrators} arbitrators.",
                "fopl": "forall x (Arbitration(x) -> Panel(x, count={arbitrators}))",
                "predicates": ["Arbitration(x)", "Panel(x, count)"],
                "type": "dispute_resolution",
                "variables": {"x": "Arbitration"},
                "params": {"arbitrators": [1, 3, 5]}
            },
            {
                "text": "Small claims under {amount} USD resolved by summary judgment.",
                "fopl": "forall x (Claim(x) & Amount(x, value < {amount}) -> SummaryJudgment(x))",
                "predicates": ["Claim(x)", "Amount(x, value)", "SummaryJudgment(x)"],
                "type": "dispute_resolution",
                "variables": {"x": "Claim"},
                "params": {"amount": [1000, 5000, 10000, 25000, 50000]}
            },
            
            # ========== PERFORMANCE OBLIGATIONS (8 variations) ==========
            {
                "text": "Project milestone must be completed within {weeks} weeks.",
                "fopl": "forall x (Milestone(x) -> Complete(x, deadline <= {weeks}))",
                "predicates": ["Milestone(x)", "Complete(x, deadline)"],
                "type": "performance",
                "variables": {"x": "Milestone"},
                "params": {"weeks": [1, 2, 4, 6, 8, 12, 16]}
            },
            {
                "text": "Service level agreement guarantees {percent}% uptime.",
                "fopl": "forall x (Service(x) -> GuaranteeUptime(x, percentage >= {percent}))",
                "predicates": ["Service(x)", "GuaranteeUptime(x, percentage)"],
                "type": "performance",
                "variables": {"x": "Service"},
                "params": {"percent": [95, 98, 99, 99.5, 99.9, 99.99]}
            },
            {
                "text": "Response time must not exceed {hours} hours for critical issues.",
                "fopl": "forall x (CriticalIssue(x) -> ResponseTime(x, hours <= {hours}))",
                "predicates": ["CriticalIssue(x)", "ResponseTime(x, hours)"],
                "type": "performance",
                "variables": {"x": "Issue"},
                "params": {"hours": [1, 2, 4, 8, 12, 24]}
            },
            {
                "text": "Minimum {units} units must be produced per month.",
                "fopl": "forall x (Producer(x) -> ProduceMinimum(x, units >= {units}))",
                "predicates": ["Producer(x)", "ProduceMinimum(x, units)"],
                "type": "performance",
                "variables": {"x": "Producer"},
                "params": {"units": [100, 500, 1000, 5000, 10000, 50000]}
            },
            {
                "text": "Quality standards require {percent}% defect-free rate.",
                "fopl": "forall x (Product(x) -> QualityStandard(x, defect_free >= {percent}))",
                "predicates": ["Product(x)", "QualityStandard(x, defect_free)"],
                "type": "performance",
                "variables": {"x": "Product"},
                "params": {"percent": [95, 97, 98, 99, 99.5, 99.9]}
            },
            {
                "text": "Contractor must provide {updates} progress updates per week.",
                "fopl": "forall x (Contractor(x) -> ProvideUpdates(x, frequency >= {updates}))",
                "predicates": ["Contractor(x)", "ProvideUpdates(x, frequency)"],
                "type": "performance",
                "variables": {"x": "Contractor"},
                "params": {"updates": [1, 2, 3, 5, 7]}
            },
            {
                "text": "Testing phase requires minimum {days} days before deployment.",
                "fopl": "forall x (Software(x) -> TestingPeriod(x, days >= {days}))",
                "predicates": ["Software(x)", "TestingPeriod(x, days)"],
                "type": "performance",
                "variables": {"x": "Software"},
                "params": {"days": [7, 14, 21, 30, 45, 60]}
            },
            {
                "text": "Performance review conducted every {months} months.",
                "fopl": "forall x (Employee(x) -> PerformanceReview(x, interval <= {months}))",
                "predicates": ["Employee(x)", "PerformanceReview(x, interval)"],
                "type": "performance",
                "variables": {"x": "Employee"},
                "params": {"months": [3, 6, 12]}
            },
            
            # ========== RENEWAL & EXTENSION (6 variations) ==========
            {
                "text": "Contract auto-renews unless terminated {days} days prior.",
                "fopl": "forall x (Contract(x) & ~Terminated(x, notice >= {days}) -> AutoRenew(x))",
                "predicates": ["Contract(x)", "Terminated(x, notice)", "AutoRenew(x)"],
                "type": "renewal",
                "variables": {"x": "Contract"},
                "params": {"days": [30, 60, 90, 120]}
            },
            {
                "text": "Renewal requires mutual written agreement {days} days in advance.",
                "fopl": "forall x (Contract(x) -> RenewalRequires(x, agreement & notice >= {days}))",
                "predicates": ["Contract(x)", "RenewalRequires(x, agreement, notice)"],
                "type": "renewal",
                "variables": {"x": "Contract"},
                "params": {"days": [30, 45, 60, 90]}
            },
            {
                "text": "Extension fee of {amount} USD applies for each additional month.",
                "fopl": "forall x (Contract(x) & Extended(x) -> PayFee(x, per_month={amount}))",
                "predicates": ["Contract(x)", "Extended(x)", "PayFee(x, per_month)"],
                "type": "renewal",
                "variables": {"x": "Contract"},
                "params": {"amount": [50, 100, 250, 500, 1000]}
            },
            {
                "text": "Renewal price increased by {percent}% annually.",
                "fopl": "forall x (Contract(x) & Renewed(x) -> PriceIncrease(x, rate={percent}))",
                "predicates": ["Contract(x)", "Renewed(x)", "PriceIncrease(x, rate)"],
                "type": "renewal",
                "variables": {"x": "Contract"},
                "params": {"percent": [3, 5, 7, 10, 15]}
            },
            {
                "text": "Maximum {renewals} renewals permitted under same terms.",
                "fopl": "forall x (Contract(x) -> MaxRenewals(x, count <= {renewals}))",
                "predicates": ["Contract(x)", "MaxRenewals(x, count)"],
                "type": "renewal",
                "variables": {"x": "Contract"},
                "params": {"renewals": [1, 2, 3, 5, 10]}
            },
            {
                "text": "Early renewal discount of {percent}% if renewed {days} days early.",
                "fopl": "forall x (Contract(x) & RenewEarly(x, days >= {days}) -> Discount(x, rate={percent}))",
                "predicates": ["Contract(x)", "RenewEarly(x, days)", "Discount(x, rate)"],
                "type": "renewal",
                "variables": {"x": "Contract"},
                "params": {"percent": [5, 10, 15, 20], "days": [30, 60, 90, 120]}
            },
            
            # ========== DATA PROTECTION (5 variations) ==========
            {
                "text": "Personal data must be deleted within {days} days of request.",
                "fopl": "forall x (DataController(x) & DeleteRequest(x) -> DeleteData(x, deadline <= {days}))",
                "predicates": ["DataController(x)", "DeleteRequest(x)", "DeleteData(x, deadline)"],
                "type": "data_protection",
                "variables": {"x": "Controller"},
                "params": {"days": [7, 14, 30, 45, 60]}
            },
            {
                "text": "Data breach notification required within {hours} hours.",
                "fopl": "forall x (Organization(x) & DataBreach(x) -> NotifyAuthorities(x, deadline <= {hours}))",
                "predicates": ["Organization(x)", "DataBreach(x)", "NotifyAuthorities(x, deadline)"],
                "type": "data_protection",
                "variables": {"x": "Organization"},
                "params": {"hours": [24, 48, 72, 96]}
            },
            {
                "text": "Data retention limited to {years} years after service ends.",
                "fopl": "forall x (Data(x) & ServiceEnded(x) -> RetentionLimit(x, years <= {years}))",
                "predicates": ["Data(x)", "ServiceEnded(x)", "RetentionLimit(x, years)"],
                "type": "data_protection",
                "variables": {"x": "Data"},
                "params": {"years": [1, 2, 3, 5, 7, 10]}
            },
            {
                "text": "Encryption required for data at rest and in transit.",
                "fopl": "forall x (SensitiveData(x) -> EncryptAtRest(x) & EncryptInTransit(x))",
                "predicates": ["SensitiveData(x)", "EncryptAtRest(x)", "EncryptInTransit(x)"],
                "type": "data_protection",
                "variables": {"x": "Data"},
                "params": {}
            },
            {
                "text": "GDPR compliance penalty up to {amount} EUR for violations.",
                "fopl": "forall x (Violation(x) & GDPRBreach(x) -> Penalty(x, max={amount}))",
                "predicates": ["Violation(x)", "GDPRBreach(x)", "Penalty(x, max)"],
                "type": "data_protection",
                "variables": {"x": "Violation"},
                "params": {"amount": [10000, 50000, 100000, 500000, 1000000]}
            },
        ]
    
    def _add_text_variation(self, clause_text: str) -> str:
        """Add linguistic variations to clause text for more diversity"""
        variations = {
            "must": ["shall", "must", "is required to", "is obligated to"],
            "shall": ["shall", "must", "will", "is required to"],
            "may": ["may", "can", "is permitted to", "has the right to"],
            "within": ["within", "in", "no later than", "before the end of"],
            "by": ["by", "before", "no later than", "on or before"],
            "after": ["after", "following", "subsequent to", "upon completion of"],
            "before": ["before", "prior to", "in advance of", "ahead of"],
            "at least": ["at least", "no less than", "minimum of", "a minimum of"],
            "no more than": ["no more than", "not exceeding", "maximum of", "up to"],
            "agrees to": ["agrees to", "consents to", "undertakes to", "commits to"],
            "responsible for": ["responsible for", "liable for", "accountable for", "in charge of"],
        }
        
        # Apply random variations (30% chance per phrase)
        for original, alternatives in variations.items():
            if original in clause_text and random.random() < 0.3:
                clause_text = clause_text.replace(original, random.choice(alternatives))
        
        return clause_text
    
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
        
        # Add linguistic variation to text (for more diversity)
        clause_text = self._add_text_variation(clause_text)
        
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
        expected_outcome = True  # Default value
        
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
            else:
                # Default payment case
                compliance_case[party] = {"Compliant": random.choice([True, False])}
                expected_outcome = compliance_case[party]["Compliant"]
                
        elif template["type"] == "termination":
            party = list(context.values())[0]
            if "days" in param_values:
                notice_given = random.choice([15, 30, 45, 60, 90])
                compliance_case[party] = {"NoticeGiven": notice_given}
                expected_outcome = notice_given >= param_values["days"]
            elif "weeks" in param_values:
                notice_weeks = random.choice([1, 2, 3, 4, 8])
                compliance_case[party] = {"NoticeWeeks": notice_weeks}
                expected_outcome = notice_weeks >= param_values["weeks"]
            elif "months" in param_values:
                notice_months = random.choice([1, 2, 3, 6])
                compliance_case[party] = {"NoticeMonths": notice_months}
                expected_outcome = notice_months >= param_values["months"]
            elif "years" in param_values:
                duration_years = random.choice([1, 2, 3, 5])
                compliance_case[party] = {"DurationYears": duration_years}
                expected_outcome = duration_years >= param_values["years"]
            else:
                compliance_case[party] = {"PaidRent": random.choice([True, False])}
                expected_outcome = not compliance_case[party]["PaidRent"]
                
        elif template["type"] == "maintenance":
            party = list(context.values())[0]
            if "months" in param_values:
                months_since = random.choice([2, 4, 6, 8, 12])
                compliance_case[party] = {"MonthsSinceMaintenance": months_since}
                expected_outcome = months_since <= param_values["months"]
            elif "days" in param_values:
                actual_days = random.choice([3, 7, 14, 20, 30])
                compliance_case[party] = {"MaintenanceDays": actual_days}
                expected_outcome = actual_days <= param_values["days"]
            elif "amount" in param_values:
                repair_cost = random.choice([50, 150, 300, 600, 1500])
                compliance_case[party] = {"RepairCost": repair_cost}
                expected_outcome = repair_cost <= param_values["amount"]
            else:
                compliance_case[party] = {"PropertyCondition": random.choice(["habitable", "poor"])}
                expected_outcome = compliance_case[party]["PropertyCondition"] == "habitable"
                
        elif template["type"] in ["access", "delivery", "warranty", "liability", "confidentiality", 
                                   "insurance", "penalty", "indemnification", "non_compete",
                                   "intellectual_property", "dispute_resolution", "performance",
                                   "renewal", "data_protection"]:
            party = list(context.values())[0]
            
            if param_values:
                param_key = list(param_values.keys())[0]
                
                if param_key == "hours":
                    actual_hours = random.choice([12, 24, 48, 72])
                    compliance_case[party] = {"Hours": actual_hours}
                    expected_outcome = actual_hours >= param_values["hours"]
                elif param_key == "days":
                    actual_days = random.choice([3, 7, 14, 20, 30, 45, 60])
                    compliance_case[party] = {"Days": actual_days}
                    expected_outcome = actual_days <= param_values["days"]
                elif param_key == "weeks":
                    actual_weeks = random.choice([1, 2, 4, 6, 8])
                    compliance_case[party] = {"Weeks": actual_weeks}
                    expected_outcome = actual_weeks <= param_values["weeks"]
                elif param_key == "months":
                    actual_months = random.choice([3, 6, 12, 24, 36])
                    compliance_case[party] = {"Months": actual_months}
                    expected_outcome = actual_months <= param_values["months"]
                elif param_key == "years":
                    actual_years = random.choice([1, 2, 3, 5, 7])
                    compliance_case[party] = {"Years": actual_years}
                    expected_outcome = actual_years <= param_values["years"]
                elif param_key == "amount":
                    actual_amount = random.choice([5000, 10000, 50000, 100000, 500000])
                    compliance_case[party] = {"Amount": actual_amount}
                    expected_outcome = actual_amount >= param_values["amount"]
                elif param_key == "percent":
                    actual_percent = random.choice([5, 10, 20, 50, 100])
                    compliance_case[party] = {"Percent": actual_percent}
                    expected_outcome = actual_percent >= param_values["percent"]
                else:
                    # Unknown parameter type
                    compliance_case[party] = {"Compliant": random.choice([True, False])}
                    expected_outcome = compliance_case[party]["Compliant"]
            else:
                # No parameters
                compliance_case[party] = {"Compliant": random.choice([True, False])}
                expected_outcome = compliance_case[party]["Compliant"]
                
        else:
            # Default case for any unhandled types
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