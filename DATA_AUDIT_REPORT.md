# Original Olist Data Audit

Generated (UTC): 2026-10-26T08:03:45+00:00

This is a read-only audit of the nine original CSV files in `data/raw/`. It does not modify or normalize source data.

| File | Rows | Columns | Key duplicate rows | SHA-256 (first 16 chars) |
|---|---:|---:|---:|---|
| `olist_customers_dataset.csv` | 99,441 | 5 | 0 | `983a422239e1712d` |
| `olist_geolocation_dataset.csv` | 1,000,163 | 5 | n/a | `b514f6fc991b9566` |
| `olist_order_items_dataset.csv` | 112,650 | 7 | 0 | `0bc4d068c4fe38cb` |
| `olist_order_payments_dataset.csv` | 103,886 | 5 | 0 | `4f713964f2815dbb` |
| `olist_order_reviews_dataset.csv` | 99,224 | 7 | 814 | `012b61c7593e34f5` |
| `olist_orders_dataset.csv` | 99,441 | 8 | 0 | `8df58ef3d2d7e994` |
| `olist_products_dataset.csv` | 32,951 | 9 | 0 | `3e6569628a17fbc7` |
| `olist_sellers_dataset.csv` | 3,095 | 4 | 0 | `1f643d2b950373b8` |
| `product_category_name_translation.csv` | 71 | 2 | 0 | `a81f0d1f27b27e72` |

## Interpretation

Multiple rows per order are expected in items, payments, reviews and geolocation. Duplicate-key counts are checks, not automatic errors. Raw files were not modified.

For the full column list, null counts, date ranges and complete hashes, see `data/processed/olist_raw_audit.json`.

## Evidence boundary

The Olist source supports historical order and delivery-outcome analysis. It does not contain observed carrier GPS scans, hub congestion logs, or comprehensive dispatch telemetry. Those signals in the project's generated benchmarks remain synthetic.
