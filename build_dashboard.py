"""
build_dashboard.py
-------------------
Regenerates dashboard/OnTime_Dashboard.html from the current contents of
dashboard/*.json and data/processed/rolling_backtest_results.csv. Run it
from the project root any time the model or data changes:

    python export_risk_summary.py    # re-scores the test set -> risk JSONs
    python build_dashboard.py        # bakes every JSON into one HTML file

The HTML file bakes in a snapshot of those files at build time, so it goes
stale exactly like the JSON it reads if this script isn't re-run. An earlier
version of the project deleted this script after first use, which is exactly
what let the dashboard drift from the model unnoticed - kept deliberately.

Also writes dashboard/_publish_fragment.html (same page without the outer
html/head/body wrapper) for hosts that add their own; it is not committed.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.join(HERE, "dashboard")


def load(name):
    with open(os.path.join(DASH, name), encoding="utf-8") as f:
        return json.load(f)


def load_backtest():
    path = os.path.join(HERE, "data", "processed", "rolling_backtest_results.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [{"month": r["test_month"], "auc": round(float(r["roc_auc"]), 3),
                 "rate": round(float(r["delay_rate"]) * 100, 1), "n": int(r["n_test"])}
                for r in csv.DictReader(f)]


DATA = {
    "eda": load("eda_summary.json"),
    "metrics": load("model_metrics.json"),
    "top_risk": load("top_risk_orders.json"),
    "fi": load("feature_importance.json"),
    "risk": load("risk_summary.json"),
    "backtest": load_backtest(),
}

HEAD = r"""<title>OnTime Dispatch Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
/* Layout: a dispatch board. Sticky header with tabs, then summary tiles,
   then the charts and the action queue a coordinator works from. */
:root{
  --bg:#eef1f4; --panel:#ffffff; --ink:#18212b; --muted:#5d6876; --line:#d9dee5;
  --accent:#b8640f; --accent-soft:#f6e7d6; --chart-1:#2f5d7c; --chart-2:#8a9bab;
  --low:#2e8b57; --med:#c9a227; --high:#d9771e; --crit:#c0392b; --on-sem:#ffffff;
  --display:"Barlow Condensed","Arial Narrow",Arial,sans-serif;
  --body:"Source Sans 3","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#11161c; --panel:#192029; --ink:#e6ebf0; --muted:#98a4b1; --line:#2b3540;
    --accent:#e89a4a; --accent-soft:#3a2a1a; --chart-1:#6fa3c7; --chart-2:#56677a;
    --low:#3fae6f; --med:#d8b443; --high:#e88a3a; --crit:#e05a4a; --on-sem:#11161c;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#11161c; --panel:#192029; --ink:#e6ebf0; --muted:#98a4b1; --line:#2b3540;
  --accent:#e89a4a; --accent-soft:#3a2a1a; --chart-1:#6fa3c7; --chart-2:#56677a;
  --low:#3fae6f; --med:#d8b443; --high:#e88a3a; --crit:#e05a4a; --on-sem:#11161c;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.45}
header{position:sticky;top:env(safe-area-inset-top,0px);z-index:10;background:var(--panel);border-bottom:1px solid var(--line);padding:12px 16px}
.bar{max-width:1140px;margin:0 auto;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px 20px}
.brand{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.brand h1{font-family:var(--display);font-weight:700;font-size:26px;letter-spacing:.02em;margin:0;text-transform:uppercase}
.brand h1 span{color:var(--accent)}
.brand p{margin:0;color:var(--muted);font-size:13px}
nav{display:flex;gap:4px;flex-wrap:wrap}
nav button{font:600 14px var(--body);color:var(--muted);background:none;border:0;border-bottom:2px solid transparent;padding:8px 10px;cursor:pointer}
nav button:hover{color:var(--ink)}
nav button.active{color:var(--ink);border-bottom-color:var(--accent)}
nav button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
main{max-width:1140px;margin:0 auto;padding-block:22px 48px;padding-inline:16px}
.page{display:none;flex-direction:column;gap:18px}
.page.active{display:flex}
.lede{font-size:16px;color:var(--muted);max-width:70ch;margin:0}
.lede b{color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}
.tile{background:var(--panel);padding:14px 16px;display:flex;flex-direction:column;gap:2px}
.tile .k{font:600 12px var(--body);text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.tile .v{font-family:var(--display);font-weight:600;font-size:38px;line-height:1.05;font-variant-numeric:tabular-nums}
.tile .s{font-size:13px;color:var(--muted)}
.tile.hl .v{color:var(--accent)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:18px}
.panel{background:var(--panel);border:1px solid var(--line);padding:16px;min-width:0}
.panel h2{font-family:var(--display);font-weight:600;font-size:20px;letter-spacing:.02em;margin:0 0 4px;text-wrap:balance}
.panel .sub{color:var(--muted);font-size:13px;margin:0 0 12px}
.chart{position:relative;height:280px}
.chart.tall{height:340px}
.note{font-size:13px;color:var(--muted);margin:10px 0 0}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{font:600 11.5px var(--body);text-transform:uppercase;letter-spacing:.07em;color:var(--muted);text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top;font-variant-numeric:tabular-nums}
td.id{font-family:var(--mono);font-size:12.5px;white-space:nowrap}
td.act{min-width:260px}
td.num,th.num{text-align:right;white-space:nowrap}
.pill{display:inline-block;font:700 11px var(--body);letter-spacing:.06em;padding:2px 8px;color:var(--on-sem);white-space:nowrap}
.pill.LOW{background:var(--low)}.pill.MEDIUM{background:var(--med)}.pill.HIGH{background:var(--high)}.pill.CRITICAL{background:var(--crit)}
.urgent{font-weight:700;color:var(--crit)}
.outcome{font-size:12.5px;white-space:nowrap}
.outcome.late{color:var(--crit)} .outcome.ok{color:var(--low)}
tr.sel td{background:var(--accent-soft);font-weight:600}
.bands{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
@media(max-width:640px){.bands{grid-template-columns:repeat(2,minmax(0,1fr))}}
.band{border-top:4px solid;padding:10px 4px 0}
.band.LOW{border-color:var(--low)}.band.MEDIUM{border-color:var(--med)}.band.HIGH{border-color:var(--high)}.band.CRITICAL{border-color:var(--crit)}
.band .name{font:700 12px var(--body);letter-spacing:.08em}
.band .big{font-family:var(--display);font-size:34px;font-weight:600;line-height:1.1;font-variant-numeric:tabular-nums}
.band .meta{font-size:12.5px;color:var(--muted)}
.meter{height:6px;background:var(--line);margin-top:6px}
.meter i{display:block;height:100%}
.band.LOW .meter i{background:var(--low)}.band.MEDIUM .meter i{background:var(--med)}.band.HIGH .meter i{background:var(--high)}.band.CRITICAL .meter i{background:var(--crit)}
.actions{display:flex;flex-direction:column;gap:10px;margin:0;padding:0;list-style:none}
.actions li{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 12px;align-items:baseline}
.actions .t{font-size:14px}
.actions .c{font-family:var(--mono);font-size:13px;color:var(--muted)}
.actions .track{grid-column:1/-1;height:5px;background:var(--line)}
.actions .track i{display:block;height:100%;background:var(--accent)}
footer{max-width:1140px;margin:0 auto;padding-block:0 32px;padding-inline:16px;color:var(--muted);font-size:12.5px}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
</style>"""

BODY = r"""<header><div class="bar">
  <div class="brand"><h1>On<span>Time</span></h1><p>Delivery delay early warning · Olist orders, Sep 2016 – Aug 2018</p></div>
  <nav id="tabs" aria-label="Dashboard pages"></nav>
</div></header>
<main>
  <section class="page" id="page-overview"></section>
  <section class="page" id="page-risk"></section>
  <section class="page" id="page-cause"></section>
  <section class="page" id="page-model"></section>
</main>
<footer>Built by build_dashboard.py from dashboard/*.json and data/processed/. Every figure comes from the project's own analysis. Test-set figures use a stratified 80/20 split; the threshold was chosen on training data only.</footer>
<script>
const DATA = __DATA_JSON__;
const {eda, metrics, top_risk: topRisk, fi, risk, backtest} = DATA;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = n => Number(n).toLocaleString("en-IN");
const pct = (x, d=1) => (x*100).toFixed(d) + "%";

const pages = [["overview","Overview"],["risk","Action queue"],["cause","Root causes"],["model","Model"]];
const tabs = document.getElementById("tabs");
function show(id){
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("active", p.id === "page-"+id));
  tabs.querySelectorAll("button").forEach(b => b.classList.toggle("active", b.dataset.id === id));
  try { localStorage.setItem("ontime-tab", id); } catch(e) {}
}
pages.forEach(([id,label]) => {
  const b = document.createElement("button"); b.textContent = label; b.dataset.id = id;
  b.onclick = () => show(id); tabs.appendChild(b);
});
let start = "overview";
try { start = localStorage.getItem("ontime-tab") || start; } catch(e) {}
if (location.hash && pages.some(p => "#"+p[0] === location.hash)) start = location.hash.slice(1);

Chart.defaults.font.family = css("--body");
Chart.defaults.color = css("--muted");
Chart.defaults.borderColor = css("--line");
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.plugins.legend.display = false;

const LABELS = {
  purchase_month:"Month of purchase", n_distinct_sellers:"Sellers in the order", promised_delivery_days:"Promised delivery window",
  seller_historical_delay_rate:"Seller's past delay rate", customer_state_historical_delay_rate:"Region's past delay rate",
  category_historical_delay_rate:"Category's past delay rate", seller_customer_distance_km:"Shipping distance",
  approval_delay_hours:"Approval delay", total_freight:"Freight cost", avg_freight:"Freight per item", same_state:"Same-state shipment",
  purchase_dow:"Day of week", n_items:"Items in order", total_weight_g:"Total weight"
};
const label = f => LABELS[f] || (f.startsWith("customer_state_") ? "Customer in " + f.slice(15)
  : f.startsWith("seller_state_") ? "Seller in " + f.slice(13)
  : f.startsWith("product_category_") ? "Category: " + f.slice(17).replace(/_/g," ")
  : f.replace(/_/g," "));

/* ---------- OVERVIEW ---------- */
const ov = eda.overview, sel = metrics[metrics.selected_model];
const peak = eda.monthly.reduce((a,b) => b.rate > a.rate ? b : a);
document.getElementById("page-overview").innerHTML = `
  <p class="lede">Out of <b>${fmt(ov.total_orders)}</b> delivered orders, <b>${ov.delay_rate}%</b> arrived after the date promised to the customer.
  OnTime scores each order when it is placed and puts the riskiest ones in front of the logistics team while there is still time to act.</p>
  <div class="tiles">
    <div class="tile"><span class="k">Orders analysed</span><span class="v">${fmt(ov.total_orders)}</span><span class="s">delivered, with valid dates</span></div>
    <div class="tile"><span class="k">Late deliveries</span><span class="v">${ov.delay_rate}%</span><span class="s">${fmt(Math.round(ov.total_orders*ov.delay_rate/100))} orders missed the promised date</span></div>
    <div class="tile"><span class="k">Average lateness</span><span class="v">${ov.avg_delay_days} days</span><span class="s">for orders that were late</span></div>
    <div class="tile hl"><span class="k">Late orders caught</span><span class="v">${risk.late_orders_caught}%</span><span class="s">by alerting on just ${risk.flagged_share}% of orders</span></div>
  </div>
  <div class="panel"><h2>Late-delivery rate by month</h2>
    <p class="sub">Share of each month's delivered orders that arrived late. Months with fewer than 200 orders are left out.</p>
    <div class="chart"><canvas id="cMonthly"></canvas></div>
    <p class="note">The peak is ${peak.month} at ${peak.rate}%. The late Nov 2017 surge and the Feb–Mar 2018 spike are capacity and seasonal effects, not just order-level factors. This is why the model needs regular retraining (see the Model tab).</p></div>`;
new Chart(cMonthly, {type:"line",
  data:{labels:eda.monthly.map(m=>m.month), datasets:[{data:eda.monthly.map(m=>m.rate), borderColor:css("--accent"),
    backgroundColor:css("--accent-soft"), fill:true, tension:.25, pointRadius:eda.monthly.map(m=>m===peak?5:2),
    pointBackgroundColor:css("--accent")}]},
  options:{scales:{y:{beginAtZero:true, ticks:{callback:v=>v+"%"}}, x:{grid:{display:false}}},
    plugins:{tooltip:{callbacks:{label:c=>`${c.raw}% late · ${fmt(eda.monthly[c.dataIndex].count)} orders`}}}}});

/* ---------- ACTION QUEUE ---------- */
const maxLate = Math.max(...risk.bands.map(b=>b.late_rate));
const bandsHtml = risk.bands.map(b => `<div class="band ${b.level}"><div class="name">${b.level}</div>
  <div class="big">${b.late_rate}%</div><div class="meta">actually late · ${fmt(b.n)} orders</div>
  <div class="meter"><i style="width:${b.late_rate/maxLate*100}%"></i></div></div>`).join("");
const maxAct = Math.max(...risk.actions.map(a=>a.n));
const actHtml = risk.actions.map(a => `<li><span class="t">${a.action}</span><span class="c">${fmt(a.n)}</span>
  <span class="track"><i style="width:${a.n/maxAct*100}%"></i></span></li>`).join("");
const rows = topRisk.map(r => {
  const urgent = r.action.startsWith("URGENT: ");
  const act = urgent ? `<span class="urgent">URGENT</span> ${r.action.slice(8)}` : r.action;
  return `<tr><td class="id">${r.order_id}</td><td><span class="pill ${r.level}">${r.level}</span></td><td class="num">${r.prob}%</td>
  <td>${r.category.replace(/_/g," ")}</td><td>${r.seller_state} → ${r.customer_state}</td><td>${r.driver}</td><td class="act">${act}</td>
  <td class="outcome ${r.actually_late?"late":"ok"}">${r.actually_late?"Arrived late":"Arrived on time"}</td></tr>`;}).join("");
document.getElementById("page-risk").innerHTML = `
  <p class="lede">On the ${fmt(risk.n_test)} held-out test orders, the alert fired on <b>${fmt(risk.n_flagged)}</b> (${risk.flagged_share}%).
  <b>${risk.flagged_late_rate}%</b> of those really were late, against ${risk.base_late_rate}% across all orders. Each alert comes with a next step tied to its main risk driver.</p>
  <div class="panel"><h2>Does the risk band mean anything?</h2>
    <p class="sub">Share of test orders in each band that really arrived late. Bands: LOW under 30%, MEDIUM 30–60%, HIGH 60–80%, CRITICAL 80% and above.</p>
    <div class="bands">${bandsHtml}</div>
    <p class="note">A CRITICAL order is about ${(risk.bands[3].late_rate/risk.bands[0].late_rate).toFixed(0)}× as likely to be late as a LOW one. The ranking holds up on orders the model never saw.</p></div>
  <div class="panel"><h2>Highest-risk orders and what to do</h2>
    <p class="sub">The 12 highest-scoring test orders, as a coordinator would see them. The last column shows what really happened.</p>
    <div class="scroll"><table><thead><tr><th>Order</th><th>Band</th><th class="num">Risk</th><th>Category</th><th>Route (UF)</th><th>Main driver</th><th>Suggested action</th><th>Outcome</th></tr></thead>
    <tbody>${rows}</tbody></table></div></div>
  <div class="panel"><h2>Actions across all ${fmt(risk.n_flagged)} alerts</h2>
    <p class="sub">The action is picked from each order's top TreeSHAP driver. ${fmt(risk.n_urgent)} CRITICAL alerts are marked URGENT.</p>
    <ul class="actions">${actHtml}</ul></div>`;

/* ---------- ROOT CAUSES ---------- */
const hbar = (el, labels, data, color, unit="%") => new Chart(el, {type:"bar",
  data:{labels, datasets:[{data, backgroundColor:color, borderRadius:0, barThickness:16}]},
  options:{indexAxis:"y", scales:{x:{beginAtZero:true, ticks:{callback:v=>v+unit}}, y:{grid:{display:false}}}}});
const maxDrv = Math.max(...risk.top_drivers.map(d=>d.n));
document.getElementById("page-cause").innerHTML = `
  <p class="lede">Lateness clusters by geography and by season more than by product. Long routes into the North-East, and orders placed during demand spikes, carry the most risk.</p>
  <div class="grid2">
    <div class="panel"><h2>Late rate by customer state</h2><p class="sub">Top 10 states with at least 100 orders.</p><div class="chart tall"><canvas id="cState"></canvas></div></div>
    <div class="panel"><h2>Late rate by product category</h2><p class="sub">Top 10 categories with at least 150 orders.</p><div class="chart tall"><canvas id="cCat"></canvas></div></div>
  </div>
  <div class="grid2">
    <div class="panel"><h2>Late rate by seller–customer distance</h2><p class="sub">Straight-line distance between zip-code centroids.</p><div class="chart"><canvas id="cDist"></canvas></div></div>
    <div class="panel"><h2>What drives each alert</h2><p class="sub">Top driver of each of the ${fmt(risk.n_flagged)} flagged test orders.</p><div class="chart"><canvas id="cDrv"></canvas></div></div>
  </div>`;
hbar(cState, eda.state.map(s=>s.state), eda.state.map(s=>s.rate), css("--chart-1"));
hbar(cCat, eda.category.map(c=>c.category.replace(/_/g," ")), eda.category.map(c=>c.rate), css("--chart-1"));
new Chart(cDist, {type:"bar", data:{labels:eda.distance.map(d=>d.bucket), datasets:[{data:eda.distance.map(d=>d.rate),
  backgroundColor:eda.distance.map((d,i,a)=>i===a.length-1?css("--accent"):css("--chart-2"))}]},
  options:{scales:{y:{beginAtZero:true, ticks:{callback:v=>v+"%"}}, x:{grid:{display:false}}}}});
hbar(cDrv, risk.top_drivers.map(d=>d.driver), risk.top_drivers.map(d=>d.n), css("--accent"), "");

/* ---------- MODEL ---------- */
const order = ["logistic_regression","decision_tree","random_forest","random_forest_tuned","xgboost","xgboost_tuned"].filter(k=>metrics[k]);
const nice = k => k.replace(/_/g," ").replace("xgboost","XGBoost").replace(/^./, c=>c.toUpperCase());
const mrows = order.map(k => { const m = metrics[k], s = k === metrics.selected_model;
  return `<tr class="${s?"sel":""}"><td>${nice(k)}${s?" (selected)":""}</td><td class="num">${m.precision.toFixed(3)}</td>
  <td class="num">${m.recall.toFixed(3)}</td><td class="num">${m.f1.toFixed(3)}</td><td class="num">${m.roc_auc.toFixed(3)}</td></tr>`;}).join("");
document.getElementById("page-model").innerHTML = `
  <p class="lede">Four model families were compared on precision, recall, F1 and ROC-AUC. Accuracy is not used, because a model that predicts "on time" for every order would already be ${(100-ov.delay_rate).toFixed(1)}% accurate.</p>
  <div class="tiles">
    <div class="tile hl"><span class="k">ROC-AUC (test)</span><span class="v">${sel.roc_auc.toFixed(3)}</span><span class="s">tuned XGBoost</span></div>
    <div class="tile"><span class="k">Recall at threshold</span><span class="v">${pct(sel.recall)}</span><span class="s">of late orders flagged</span></div>
    <div class="tile"><span class="k">Precision at threshold</span><span class="v">${pct(sel.precision)}</span><span class="s">vs ${pct(metrics.base_delay_rate)} base rate</span></div>
    <div class="tile"><span class="k">Alert threshold</span><span class="v">${metrics.selected_threshold.toFixed(3)}</span><span class="s">chosen on training data only</span></div>
  </div>
  <div class="panel"><h2>Model comparison</h2><p class="sub">${fmt(metrics.n_train)} training orders, ${fmt(metrics.n_test)} test orders.</p>
    <div class="scroll"><table><thead><tr><th>Model</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">ROC-AUC</th></tr></thead><tbody>${mrows}</tbody></table></div></div>
  <div class="grid2">
    <div class="panel"><h2>Rolling backtest: retrain monthly</h2>
      <p class="sub">Train on everything before a month, test on that month only. Mean ROC-AUC ${metrics.rolling_backtest_mean_auc.toFixed(3)} over ${backtest.length} months.</p>
      <div class="chart"><canvas id="cBack"></canvas></div>
      <p class="note">A single fixed time split scored only ${metrics.static_time_split_auc_legacy} because delay rates shift with the season. Monthly retraining recovers most of the signal.</p></div>
    <div class="panel"><h2>Most important features</h2><p class="sub">XGBoost gain importance, top 10 (one-hot columns shown separately).</p><div class="chart"><canvas id="cFI"></canvas></div></div>
  </div>`;
new Chart(cBack, {type:"line", data:{labels:backtest.map(b=>b.month), datasets:[
  {data:backtest.map(b=>b.auc), borderColor:css("--chart-1"), backgroundColor:css("--chart-1"), pointRadius:3, tension:.2},
  {data:backtest.map(()=>0.5), borderColor:css("--muted"), borderDash:[4,4], pointRadius:0}]},
  options:{scales:{y:{min:0.4, max:0.9}, x:{grid:{display:false}}},
    plugins:{tooltip:{filter:i=>i.datasetIndex===0, callbacks:{label:c=>`AUC ${c.raw} · ${fmt(backtest[c.dataIndex].n)} test orders`}}}}});
hbar(cFI, fi.slice(0,10).map(f=>label(f.feature)), fi.slice(0,10).map(f=>f.importance), css("--chart-1"), "");

show(start);
</script>"""

body = BODY.replace("__DATA_JSON__", json.dumps(DATA, ensure_ascii=False))
full = ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + HEAD + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")

with open(os.path.join(DASH, "OnTime_Dashboard.html"), "w", encoding="utf-8") as f:
    f.write(full)
with open(os.path.join(DASH, "_publish_fragment.html"), "w", encoding="utf-8") as f:
    f.write(HEAD + "\n" + body + "\n")
print("written", len(full), "bytes")
