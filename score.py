#!/usr/bin/env python3
"""
score.py - score new orders for delivery-delay risk from the command line.

    python score.py --template                      # writes template_orders.csv
    python score.py --input my_orders.csv           # writes scored_orders.csv
    python score.py --input my_orders.csv --output out.csv

Every input row produces exactly one output row: SCORED (with a delay
probability, a LOW/MEDIUM/HIGH/CRITICAL band, a confidence flag and -
for orders over the alert threshold - the reasons driving the score) or
REJECTED (with the reasons it could not be scored). Nothing is silently
dropped or silently repaired.

The three historical delay-rate features are filled in automatically
from models/rate_lookup.json using the seller_id, product_category and
customer_state columns, so you do not need to compute them yourself.
"""

import argparse
import os
import sys
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from features import FEATURE_COLUMNS                                    # noqa: E402
from artifacts import load_model, load_json, ModelIntegrityError  # noqa: E402
from inference_validation import load_reference_stats                   # noqa: E402
from scoring import score_orders, required_input_columns, summarise     # noqa: E402

# Settings can be overridden through environment variables (see .env.example),
# which is how the Docker image is configured without editing code.
MODEL_DIR = os.environ.get("ONTIME_MODEL_DIR", os.path.join(HERE, "models"))
TOP_N = int(os.environ.get("ONTIME_TOP_REASONS", "3"))


def write_template(path, n=8):
    """A few real held-out orders (outcome columns removed) so the layout is obvious."""
    queue = pd.read_csv(os.path.join(HERE, "backend", "app_data", "queue.csv"), dtype={"seller_id": str})
    cols = ["order_id", "seller_id"] + required_input_columns()
    queue.sample(n, random_state=3)[cols].to_csv(path, index=False)
    print(f"Wrote {n} example rows to {path}")
    print("Required columns:", ", ".join(required_input_columns()))
    print("Optional: order_id, seller_id (used to look up the seller's delay history).")


def main():
    ap = argparse.ArgumentParser(description="Score new orders for delivery-delay risk.")
    ap.add_argument("--input", help="CSV of orders to score")
    ap.add_argument("--output", default="scored_orders.csv", help="where to write results")
    ap.add_argument("--template", nargs="?", const="template_orders.csv", metavar="PATH",
                    help="write an example input CSV and exit")
    args = ap.parse_args()

    if args.template:
        write_template(args.template)
        return 0
    if not args.input:
        ap.error("give --input FILE (or --template to create an example)")

    try:
        from scoring import read_orders_csv
        orders = read_orders_csv(args.input)
    except Exception as e:
        print(f"Could not read {args.input}: {e.__class__.__name__}: {e}")
        return 1

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, meta = load_model(MODEL_DIR, FEATURE_COLUMNS,
                                     data_path=os.path.join(HERE, "data", "processed", "ontime_model_ready.csv"))
        for w in caught:
            print(f"WARNING: {w.message}")
        stats = load_reference_stats(os.path.join(MODEL_DIR, "reference_stats.json"))
        lookup = load_json(os.path.join(MODEL_DIR, "rate_lookup.json"))
    except (ModelIntegrityError, FileNotFoundError, RuntimeError) as e:
        print(f"Cannot load the model: {e}")
        return 1

    try:
        scored = score_orders(orders, model, meta["threshold"], stats, lookup, top_n=TOP_N)
    except ValueError as e:
        print(f"Input rejected: {e}")
        return 1

    scored.to_csv(args.output, index=False)
    print(summarise(scored))
    print(f"Alert threshold: {meta['threshold']:.3f}   Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
