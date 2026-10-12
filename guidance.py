"""Recycling guidance layer.

The AI model ONLY predicts an image class (e.g. "Plastic").
This file turns that class into disposal advice. It is kept separate so you can
edit the advice without touching any model code.

Every class has:
  item      - friendly name shown to the user
  category  - broader material group
  route     - where the item generally goes (used for the dashboard)
  recyclable- "yes" (generally recyclable), "conditional", "no", or "n/a"
  guidance  - practical, safe instructions
  warning   - why the answer may differ in your area
"""

# --- Settings for deciding when the model is too unsure -----------------------
MIN_CONFIDENCE = 0.60   # top prediction below this -> "low confidence"
MIN_MARGIN = 0.15       # top minus second prediction below this -> "low confidence"

# --- Routes (the "where does it go" bucket) ------------------------------------
ROUTE_RECYCLE = "Recycle"
ROUTE_COMPOST = "Compost / organics"
ROUTE_SPECIAL = "Special drop-off"
ROUTE_REUSE = "Donate / reuse"
ROUTE_GENERAL = "General waste"
ROUTE_REVIEW = "Needs review"

LOCAL_RULES_WARNING = "Local recycling rules vary by city and collector. Check your local rules before sorting."

# Keys must match the class names the model outputs (see classifier.EXPECTED_LABELS).
GUIDANCE = {
    "Battery": {
        "item": "Battery",
        "category": "Electronic waste / batteries",
        "route": ROUTE_SPECIAL,
        "recyclable": "no",
        "guidance": (
            "Never put batteries in the household recycling bin or general waste. "
            "Take them to an authorised battery or e-waste collection point, or a shop that accepts used batteries. "
            "Tape over the terminals of lithium or button batteries and keep them dry until drop-off."
        ),
        "warning": "Damaged, swollen or hot batteries are a fire risk: keep them away from heat and ask a collection point how to hand them over safely. " + LOCAL_RULES_WARNING,
    },
    "Biological": {
        "item": "Food / organic waste",
        "category": "Organic waste",
        "route": ROUTE_COMPOST,
        "recyclable": "no",
        "guidance": (
            "Use a food-waste collection or a home or community compost if one is available. "
            "Keep it free of plastic, packaging and stickers."
        ),
        "warning": "Many composting systems do not accept meat, dairy, cooked food or oils. " + LOCAL_RULES_WARNING,
    },
    "Cardboard": {
        "item": "Cardboard",
        "category": "Paper & cardboard",
        "route": ROUTE_RECYCLE,
        "recyclable": "conditional",
        "guidance": (
            "Clean, dry cardboard is generally recyclable. Remove tape and packing material and flatten boxes."
        ),
        "warning": "Greasy or food-soiled cardboard (such as a used pizza box), wet cardboard and wax-coated boxes may not be accepted. " + LOCAL_RULES_WARNING,
    },
    "Clothes": {
        "item": "Clothing / textiles",
        "category": "Textiles",
        "route": ROUTE_REUSE,
        "recyclable": "no",
        "guidance": (
            "Wearable clothes can be donated or reused. Worn-out textiles may be accepted at a textile collection point. "
            "Do not put clothing in the paper, plastic or glass recycling bins."
        ),
        "warning": "Wet, mouldy or heavily soiled textiles usually cannot be donated or recycled. " + LOCAL_RULES_WARNING,
    },
    "Glass": {
        "item": "Glass",
        "category": "Glass",
        "route": ROUTE_RECYCLE,
        "recyclable": "conditional",
        "guidance": (
            "Empty and rinse glass bottles and jars. Check whether lids should be removed. "
            "Wrap broken glass safely in paper before disposal."
        ),
        "warning": "This advice is for bottles and jars. Mirrors, window glass, drinking glasses, ceramics and oven-proof glass are usually NOT accepted with bottle and jar recycling. " + LOCAL_RULES_WARNING,
    },
    "Metal": {
        "item": "Metal",
        "category": "Metal",
        "route": ROUTE_RECYCLE,
        "recyclable": "conditional",
        "guidance": (
            "Empty and rinse food and drink cans and tins. Be careful of sharp edges. "
            "Clean aluminium foil can be recycled by some collectors."
        ),
        "warning": "Aerosol cans, paint tins, batteries and electronic parts need separate handling. " + LOCAL_RULES_WARNING,
    },
    "Paper": {
        "item": "Paper",
        "category": "Paper & cardboard",
        "route": ROUTE_RECYCLE,
        "recyclable": "conditional",
        "guidance": (
            "Clean, dry paper such as office paper, newspapers and magazines is generally recyclable. Keep it dry and separate from food waste."
        ),
        "warning": "Food-soiled paper, tissues, paper towels, thermal receipts, laminated or plastic-coated paper and wax paper are often NOT recyclable. " + LOCAL_RULES_WARNING,
    },
    "Plastic": {
        "item": "Plastic",
        "category": "Plastic",
        "route": ROUTE_RECYCLE,
        "recyclable": "conditional",
        "guidance": (
            "Check the resin code (the number in the triangle) and your local list of accepted plastics. "
            "If accepted, empty and rinse the item before recycling."
        ),
        "warning": "Not all plastic is recyclable. Plastic bags, films, multilayer packaging (such as snack wrappers), polystyrene foam and dirty plastic are often NOT accepted in household recycling. " + LOCAL_RULES_WARNING,
    },
    "Shoes": {
        "item": "Shoes / footwear",
        "category": "Footwear",
        "route": ROUTE_REUSE,
        "recyclable": "no",
        "guidance": (
            "Wearable shoes can be donated or reused. Tie pairs together so they stay matched. "
            "Worn-out shoes are usually not accepted in household recycling bins; ask about a footwear take-back scheme."
        ),
        "warning": "Most shoes are made from mixed materials that kerbside recycling cannot process. " + LOCAL_RULES_WARNING,
    },
    "Trash": {
        "item": "General waste",
        "category": "Non-recyclable / mixed",
        "route": ROUTE_GENERAL,
        "recyclable": "no",
        "guidance": (
            "This looks like general (residual) waste. Put it in the general waste bin. "
            "If the item is made of several materials, separating clean recyclable parts first can help."
        ),
        "warning": "The model has a single catch-all class for items that do not fit the other nine classes, so check whether part of the item belongs in another stream. " + LOCAL_RULES_WARNING,
    },
}

STATUS_CLASSIFIED = "classified"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_UNSUPPORTED = "unsupported"


def interpret(predictions):
    """Turn model predictions into a result for the user.

    predictions: list of {"label": str, "probability": float 0-1}, best first.
    Returns a dict. Uncertain predictions are NOT turned into confident advice.
    """
    if not predictions:
        raise ValueError("predictions must not be empty")

    top = predictions[0]
    label = top["label"]
    confidence = float(top["probability"])
    second = float(predictions[1]["probability"]) if len(predictions) > 1 else 0.0
    alternatives = [
        {"label": p["label"], "probability": float(p["probability"])} for p in predictions[1:]
    ]

    base = {
        "predicted_label": label,
        "confidence": confidence,
        "alternatives": alternatives,
    }

    info = GUIDANCE.get(label)
    if info is None:
        # The model returned a class we have no advice for. Be honest about it.
        return {
            **base,
            "status": STATUS_UNSUPPORTED,
            "item": label,
            "category": "Unsupported item",
            "route": ROUTE_REVIEW,
            "recyclable": "n/a",
            "guidance": "ECHO SCAN has no disposal advice for this prediction. Please check your local waste rules.",
            "warning": LOCAL_RULES_WARNING,
            "message": "The model predicted a class this app does not support.",
        }

    if confidence < MIN_CONFIDENCE or (confidence - second) < MIN_MARGIN:
        return {
            **base,
            "status": STATUS_LOW_CONFIDENCE,
            "item": "Not sure",
            "category": "Uncertain",
            "route": ROUTE_REVIEW,
            "recyclable": "n/a",
            "guidance": (
                "The model is not confident about this image, so ECHO SCAN will not give a sorting recommendation. "
                "Retake the photo with one item, in good light, against a plain background, and scan again. "
                "If it is still unclear, check your local waste rules."
            ),
            "warning": LOCAL_RULES_WARNING,
            "message": (
                f"Closest guess: {info['item']} ({confidence * 100:.0f}%). "
                "This is too uncertain to rely on."
            ),
        }

    return {
        **base,
        "status": STATUS_CLASSIFIED,
        "item": info["item"],
        "category": info["category"],
        "route": info["route"],
        "recyclable": info["recyclable"],
        "guidance": info["guidance"],
        "warning": info["warning"],
        "message": "",
    }
