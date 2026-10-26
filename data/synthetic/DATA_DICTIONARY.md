# Simulated tracking data set (companion to Olist)

**SIMULATED.** Real Olist does not publish carrier scans, failed delivery attempts, weather or strikes.
This folder is invented data built for the same orders, to show what OnTime could do if such data existed.
Never quote a result on it as a result on real Olist.

## What is real and what is invented
| Real (from Olist) | Invented (seed 2026) |
|---|---|
| order_id, purchase time, carrier pickup time (PICKED_UP), delivery time (DELIVERED), promised date, states | hub scans, out-for-delivery, failed attempts, weather calendar, strike calendar |

## Rules (simulation version 2: strengthened once, at the owner's request, after version 1 gave ROC-AUC 0.87-0.89)
* Scan times lie between the real pickup and the real delivery. Number of hub scans grows with distance (1 to 5 hubs).
* **Stall:** 70% of late orders and 8% of on-time orders get their scans bunched early, leaving a long silent gap.
* **Failed attempt:** 30% of late and 3% of on-time orders get one failed delivery attempt (a second one for 25% of those late ones).
* **Dropped scans:** each hub scan is lost with probability 8% (real feeds have gaps).
* **Weather:** per state and day, severity 0 to 3. Bad weather is more likely in Nov-Mar and in state-weeks whose real late rate was above average.
* **Carrier:** every order gets one of 5 carriers (C1 to C5). Share of late orders by carrier C1..C5: 5/10/20/30/35%; of on-time orders: 35/28/20/12/5%.
* **Hub congestion:** per state and day, an index from 0 to 1 that is higher on Mondays and Tuesdays, in Nov-Dec, and in state-weeks with an above-average real late rate.
* **Road disruptions:** per state, 1 to 4 days (landslide, flooded road, bridge closed, truck accident), more likely in state-weeks with an above-average real late rate.
* **Holidays:** the real Brazilian national holidays and carnival days for 2016 to 2018 (this file is REAL, not simulated).
* **Strikes:** per state and week, more likely when the real late rate of that state-week was above average; last 2 to 5 days.

## Files
* `tracking_events_simulated.csv`: order_id, event_time, event_type (PICKED_UP, ARRIVED_HUB, DEPARTED_HUB, OUT_FOR_DELIVERY, DELIVERY_ATTEMPT_FAILED, DELIVERED), hub_state, carrier, simulated.
* `weather_daily_simulated.csv`: date, state, severity (0 none, 1 rain, 2 heavy rain, 3 storm/flood), rain_mm, simulated.
* `strikes_simulated.csv`: strike_id, state, start_date, end_date, kind, simulated.
* `hub_congestion_daily_simulated.csv`: date, state, index (0 to 1), simulated.
* `road_disruptions_simulated.csv`: state, start_date, end_date, kind, simulated.
* `holidays_brazil.csv`: date, name, simulated (0 = real).
