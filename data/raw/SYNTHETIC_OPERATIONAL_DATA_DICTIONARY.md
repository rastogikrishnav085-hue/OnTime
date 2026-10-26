# Synthetic operational orders CSV — data dictionary

File: `data/raw/synthetic_operational_orders.csv`
Rows: 80,000 (plus header). This is intentionally a separate CSV, not appended to any observed Olist CSV.

## Relationship to the supplied raw CSVs
- Uses Olist's order, item, customer, seller, and product table schemas as the base.
- Customer/seller/product IDs and commercial/product attributes are sampled from the corresponding supplied Olist CSVs.
- Synthetic order IDs (`SYNTH_...`) deliberately do not collide with or pretend to join to observed `olist_orders_dataset.csv` orders.
- Canonical Olist-style order/item/customer/seller/product columns are included for convenient schema inspection; operational simulation fields are appended.

## Important columns
- `order_id`: synthetic-only identifier.
- `order_status`: `synthetic_simulated`; not an observed Olist status.
- `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_estimated_delivery_date`: simulated operational timeline; `order_delivered_customer_date` is blank to avoid encoding future outcomes.
- `route_km`, `weather_severity_forecast`, `congestion_forecast`, `carrier_reliability_prior`, `capacity_utilization_at_dispatch`, `remaining_sla_slack_hours`, `facility_backlog_at_dispatch`, `route_disruption_alert_at_dispatch`, `last_scan_age_hours`, `estimated_truck_delay_hours_at_dispatch`, `hour_at_dispatch`, `weekday_at_dispatch`, `dispatch_lag_hours`: dispatch-time features available to the benchmark.
- `synthetic_late_delivery`: simulated binary target (1 = simulated late, 0 = simulated on time). This is synthetic ground truth, not a label observed in Olist. Never use as an input feature.
- `record_source`: provenance marker.

## Modeling controls
Training uses only the 16 documented numeric dispatch-time features listed in `models/synthetic_operational/metrics.json`. IDs, raw status/timestamps, the target, and future customer-delivery timestamp are excluded. Records are ordered by simulated purchase time and split 70% train / 15% validation / 15% untouched chronological test.

## Limitations
The operational risk factors and target are simulated. Olist values anchor realistic entity/category/commercial distributions but do not make the synthetic labels observed facts. Synthetic benchmark metrics must not be reported as real-world Olist model performance.
