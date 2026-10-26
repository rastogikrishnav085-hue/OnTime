"""
categories.py
--------------
The project focuses on six product categories instead of all 70+.

Why (mentor's guidance, review of 2 Oct 2026): pick 5-6 product types and
be sharp about them, because a late delivery means different things for
different products - a late baby or health product is a real problem for
the customer, a late cushion cover is an annoyance. Every other category
is still scored by the model, but the category-specific analysis and
playbooks below cover these six, and the rest is listed as roadmap.

`tier` is a business judgement, not something the data says. The weights
feed one optional sort order in the app ("priority = risk x weight") and
are kept here, in one place, so they are easy to change or switch off.
"""

FOCUS_CATEGORIES = {
    "health_beauty": {
        "label": "Health & beauty",
        "tier": "Essential",
        "weight": 1.30,
        "why": "Second-largest category. Includes personal-care and health items people need on a schedule.",
        "impact": "A late health or hygiene product can mean a customer runs out of something they rely on; trust drops sharply.",
        "playbook": [
            "Treat HIGH and CRITICAL orders as same-day priority in the dispatch queue.",
            "Offer an express upgrade at checkout on routes with a long transit history.",
            "Message the customer proactively with a revised date instead of waiting for a complaint.",
        ],
    },
    "baby": {
        "label": "Baby",
        "tier": "Essential",
        "weight": 1.30,
        "why": "Highest late rate of the large categories (9.2%) and products parents need on time.",
        "impact": "Missing nappies, feeding or hygiene items cause immediate stress and very likely a lost customer.",
        "playbook": [
            "Never promise the platform's shortest window on long routes for this category.",
            "Keep fast-moving baby SKUs in a regional hub close to the North-East and South demand.",
            "Escalate any CRITICAL baby order to a person the same day.",
        ],
    },
    "watches_gifts": {
        "label": "Watches & gifts",
        "tier": "Deadline-sensitive",
        "weight": 1.20,
        "why": "Bought for birthdays and festivals, so the date matters more than the item.",
        "impact": "A gift that arrives after the occasion has lost most of its value, even if it arrives in perfect condition.",
        "playbook": [
            "Ask for the occasion date at checkout and flag orders whose promise is too close to it.",
            "Add buffer days to promises in November and before festival dates.",
            "Offer gift-message e-cards when a delay is predicted, so the customer has something on the day.",
        ],
    },
    "computers_accessories": {
        "label": "Computers & accessories",
        "tier": "High value",
        "weight": 1.15,
        "why": "High order values and often work-critical purchases.",
        "impact": "Delays hold up someone's work and lead to costly refunds and chargebacks.",
        "playbook": [
            "Insure and track high-value parcels; call the customer on CRITICAL alerts.",
            "Use the national carrier's tracked service for orders above the category's median value.",
            "Review sellers in this category with a late rate above 15% every month.",
        ],
    },
    "furniture_decor": {
        "label": "Furniture & decor",
        "tier": "Bulky",
        "weight": 1.05,
        "why": "Heavy and bulky items that need special handling and often longer routes.",
        "impact": "Bulky deliveries are hard to re-route, and failed delivery attempts add days.",
        "playbook": [
            "Route parcels above 10 kg to a carrier with a bulky-freight service.",
            "Book delivery slots with the customer instead of open-ended delivery windows.",
            "Set longer default promises for long-haul bulky orders.",
        ],
    },
    "bed_bath_table": {
        "label": "Bed, bath & table",
        "tier": "High volume",
        "weight": 1.00,
        "why": "The single largest category (9,239 delivered orders), so small fixes add up.",
        "impact": "Individually low urgency, but volume makes it the biggest source of late orders in absolute terms.",
        "playbook": [
            "Focus on peak-season capacity: most late orders here cluster in Nov and Feb-Mar.",
            "Batch dispatch from the top sellers to cut first-mile delays.",
            "Use the threshold planner to keep alert volume manageable for this category.",
        ],
    },
}

ROADMAP_NOTE = (
    "Other categories (sports & leisure, housewares, telephony, auto, toys and the rest) are scored by "
    "the same model but are outside the focused analysis. Extending the playbooks to them, and adding "
    "quick-commerce data, is on the roadmap."
)


def tier_of(category):
    c = FOCUS_CATEGORIES.get(category)
    return c["tier"] if c else ""


def weight_of(category):
    c = FOCUS_CATEGORIES.get(category)
    return c["weight"] if c else 1.0
