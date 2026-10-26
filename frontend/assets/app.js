/* OnTime frontend - plain JS, no build step. Talks to the FastAPI backend under /api. */
(() => {
"use strict";

// ------------------------------------------------------------------ helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// One-line "where did this score come from" note, built from the backend's provenance block.
const provNote = (p) => !p ? "" : `<p class="note prov ${p.synthetic ? "synthetic" : ""}"><b>${p.synthetic ? "Generated-data benchmark" : "Historical data"}</b> · stage: ${esc(p.stage)} · ${esc(p.data_origin)}. ${esc(p.score_meaning)}.</p>`;
const fmt = (n, d = 0) => n == null || isNaN(n) ? "—" : Number(n).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (p, d = 0) => p == null ? "—" : (p * 100).toFixed(d) + "%";
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const nice = (s) => String(s ?? "").replace(/_/g, " ");
const shortId = (id) => id ? id.slice(0, 10) + "…" : "—";
const BANDS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
const pill = (lvl) => lvl ? `<span class="pill ${esc(lvl)}">${esc(lvl)}</span>` : "";
const date = (s) => s ? new Date(s.replace(" ", "T")).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" }) : "—";
// compact versions for dense tables: full wording stays in the tooltip and the order drawer
const queueAction = (a) => {
  if (!a) return "";
  if (a.startsWith("URGENT")) return `<span class="urgent">URGENT</span>`;
  const head = a.split(":")[0].trim();
  return esc(head.length > 34 ? head.slice(0, 33) + "…" : head);
};
const shortDriver = (d) => (d || "").replace(/\s*\((days|hours)\)\s*$/, "").replace(/\s*\(days after purchase\)\s*$/, "");
const action = (a) => !a ? "" : a.startsWith("URGENT: ") ? `<span class="urgent">URGENT</span> ${esc(a.slice(8))}` : esc(a);
const bandColor = (lvl) => css({ LOW: "--low", MEDIUM: "--med", HIGH: "--high", CRITICAL: "--crit" }[lvl] || "--accent");

function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2600);
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {}, ...opts });
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) {
    const msg = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { } },
};

// ------------------------------------------------------------------ charts
let charts = [];
function chart(el, cfg) {
  Chart.defaults.font.family = css("--body");
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--line");
  cfg.options = Object.assign({ maintainAspectRatio: false, responsive: true, animation: { duration: 300 } }, cfg.options || {});
  cfg.options.plugins = Object.assign({ legend: { display: false } }, cfg.options.plugins || {});
  const c = new Chart(el, cfg); charts.push(c); return c;
}
const hbar = (el, labels, data, color, opts = {}) => chart(el, {
  type: "bar", data: { labels, datasets: [{ data, backgroundColor: color, borderRadius: 3, barThickness: opts.thick || 14 }] },
  options: { indexAxis: "y", scales: { x: { beginAtZero: true, ticks: { callback: v => v + (opts.unit ?? "%") } }, y: { grid: { display: false } } },
    plugins: { tooltip: { callbacks: { label: c => opts.tip ? opts.tip(c) : `${c.raw}${opts.unit ?? "%"}` } } } },
});
const vbar = (el, labels, data, colors, opts = {}) => chart(el, {
  type: "bar", data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 3 }] },
  options: { scales: { y: { beginAtZero: true, ticks: { callback: v => v + (opts.unit ?? "%") } }, x: { grid: { display: false } } },
    plugins: { tooltip: { callbacks: { label: c => opts.tip ? opts.tip(c) : `${c.raw}${opts.unit ?? "%"}` } } } },
});

// ------------------------------------------------------------------ app state
const S = { meta: null, summary: null };
const view = $("#view");

const PAGES = {
  overview: { title: "Overview", sub: "Where the risk is today, and what needs a person", render: renderOverview },
  queue: { title: "Risk queue", sub: "Every incoming order, scored at checkout", render: renderQueue },
  board: { title: "Action board", sub: "Flagged orders the team is working on", render: renderBoard },
  planner: { title: "Alert planner", sub: "Pick how many alerts to send, using your own costs", render: renderPlanner },
  simulate: { title: "Score an order", sub: "Change an order and watch the risk move", render: renderSimulate },
  batch: { title: "Batch scoring", sub: "Upload a CSV of new orders, get every one scored", render: renderBatch },
  model: { title: "Model", sub: "How well the selected model works, and how that was measured", render: renderModel },
  about: { title: "How it works", sub: "From raw orders to a suggested action", render: renderAbout },
  briefing: { title: "Briefing", sub: "The periodic check-in: what happened, why, and what to do next", render: renderBriefing },
  insights: { title: "Insights", sub: "Where and when orders arrive late", render: renderInsights },
  patterns: { title: "Delay patterns", sub: "Late orders grouped by shared cause, with what to do about each group", render: renderPatterns },
  focus: { title: "Focus categories", sub: "Six product types where a late delivery matters most", render: renderFocus },
  network: { title: "Network", sub: "Delivery models, hubs and carriers", render: renderNetwork },
  assistant: { title: "Ask OnTime", sub: "Questions answered from the app's own data", render: renderAssistant },
  literature: { title: "Literature", sub: "Related work, and where OnTime uses it", render: renderLiterature },
  adapt: { title: "Adapt to your data", sub: "Use OnTime on a data set that is not Olist", render: renderAdapt },
  settings: { title: "Settings", sub: "Appearance and the assistant's engines", render: renderSettings },
};

function parseHash() {
  const h = location.hash.replace(/^#\/?/, "");
  const [page, qs] = h.split("?");
  return { page: PAGES[page] ? page : "overview", params: new URLSearchParams(qs || "") };
}
function go(page, params) {
  const qs = params && [...params].length ? "?" + params.toString() : "";
  location.hash = `#/${page}${qs}`;
}

async function route() {
  const { page, params } = parseHash();
  charts.forEach(c => c.destroy()); charts = [];
  $$("#nav a").forEach(a => { const on = a.dataset.page === page; a.classList.toggle("active", on); on ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"); });
  $$(".tabbar a").forEach(a => { const on = a.dataset.page === page; a.classList.toggle("active", on); on ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"); });
  const _tm = document.getElementById("tab-more"); if (_tm) _tm.classList.toggle("active", !document.querySelector(`.tabbar a[data-page="${page}"]`));
  if (typeof revealActiveGroup === "function") revealActiveGroup(page);
  $("#page-title").textContent = PAGES[page].title;
  $("#page-sub").textContent = PAGES[page].sub;
  document.title = `${PAGES[page].title} · OnTime`;
  view.innerHTML = `<div class="skeleton">Loading…</div>`;
  try { await PAGES[page].render(params); }
  catch (e) { view.innerHTML = `<div class="panel"><h2>Something went wrong</h2><p class="err">${esc(e.message)}</p><p class="hint">Check that the server is running (python run.py), then reload.</p></div>`; }
}

async function refreshCounts() {
  try {
    const b = await api("/api/board");
    const open = b.counts.to_review + b.counts.investigating + b.counts.customer_informed;
    $("#nav-board").textContent = open || "";
  } catch (e) { }
}

// ------------------------------------------------------------------ board actions (shared)
async function addToBoard(orderId, btn) {
  try {
    await api("/api/board", { method: "POST", body: JSON.stringify({ order_id: orderId }) });
    toast("Added to the action board");
    if (btn) { btn.outerHTML = `<span class="tag">On board</span>`; }
    refreshCounts();
  } catch (e) { toast(e.message); }
}

// ------------------------------------------------------------------ THEMES
const THEMES = [
  { key: "dispatch", label: "Light", mode: "light", bg: "#f4f4f5", ac: "#09090b", pair: "dispatch-night" },
  { key: "dispatch-night", label: "Dark", mode: "dark", bg: "#000000", ac: "#ffffff", pair: "dispatch" },
  { key: "emerald", label: "Emerald", mode: "light", bg: "#edf3ef", ac: "#1d7a4c", pair: "forest" },
  { key: "forest", label: "Forest", mode: "dark", bg: "#0c1611", ac: "#4cc38a", pair: "emerald" },
  { key: "ocean", label: "Ocean", mode: "light", bg: "#ecf1f8", ac: "#1f66c9", pair: "navy" },
  { key: "navy", label: "Navy", mode: "dark", bg: "#0a1120", ac: "#5ea4ff", pair: "ocean" },
  { key: "paper", label: "Paper white", mode: "light", bg: "#ffffff", ac: "#141414", pair: "graphite" },
  { key: "graphite", label: "Graphite black", mode: "dark", bg: "#0a0a0a", ac: "#f2f2f2", pair: "paper" },
];
const currentTheme = () => THEMES.find(t => t.key === document.documentElement.getAttribute("data-theme")) || THEMES[0];
let themeFadeTimer = null;
function setTheme(key) {
  const root = document.documentElement;
  root.classList.add("theme-fade");                       // short colour fade; CSS turns it off for reduced motion
  clearTimeout(themeFadeTimer); themeFadeTimer = setTimeout(() => root.classList.remove("theme-fade"), 450);
  root.setAttribute("data-theme", key);
  try { localStorage.setItem("ontime-theme", key); } catch (e) { }
  applyThemeLabel(); renderRailSwatches(); route();
}
function swatchesHtml() {
  const cur = currentTheme().key;
  return `<div class="swatches" role="group" aria-label="Colour theme">${THEMES.map(t => `<button class="swatch" type="button"
    style="--sw-bg:${t.bg};--sw-ac:${t.ac}" data-theme-key="${t.key}" aria-pressed="${t.key === cur}" title="${t.label}" aria-label="${t.label} theme"></button>`).join("")}
    <span class="swatch-label">${esc(currentTheme().label)}</span></div>`;
}
function bindSwatches(root = document) {
  $$("[data-theme-key]", root).forEach(b => b.addEventListener("click", () => setTheme(b.dataset.themeKey)));
}

async function updateDbStatus() {
  const el = document.getElementById("rail-db-status");
  if (!el) return;
  try {
    const db = await api("/api/database/status");
    if (db.mode === "supabase") {
      el.innerHTML = `<span class="db-pill supabase ${db.connected ? "on" : "off"}">Cloud: Supabase ${db.connected ? "Connected" : "Offline"}</span>`;
    } else {
      el.innerHTML = `<span class="db-pill sqlite">Database: SQLite (Local)</span>`;
    }
  } catch (e) {
    el.innerHTML = `<span class="db-pill sqlite">Database: Local</span>`;
  }
}

function renderRailSwatches() {
  const el = document.getElementById("rail-swatches");
  if (!el) return;
  const cur = currentTheme().key;
  el.innerHTML = THEMES.map(t => `<button class="rail-swatch" type="button"
    style="--sw-bg:${t.bg};--sw-ac:${t.ac}" data-theme-key="${t.key}" aria-pressed="${t.key === cur}"
    aria-label="${esc(t.label)} (${t.mode})" title="${esc(t.label)} - ${t.mode}">
  </button>`).join("");
  bindSwatches(el);
  const curObj = currentTheme();
  const hdr = document.querySelector(".rail-theme-header span");
  if (hdr) hdr.innerHTML = `Theme: <b style="color:var(--rail-ink);font-weight:700;letter-spacing:0">${esc(curObj.label)}</b>`;
  updateDbStatus();
}


// A short note that appears only when the selected model changes how the numbers should be read.
function viewNotice() {
  const v = S.meta && S.meta.queue_view;
  return v && v.note ? `<div class="callout warn" style="margin:10px 0"><div class="k">${v.in_sample ? "Read with care" : "Model note"} · ${esc(v.label)}</div>${esc(v.note)}</div>` : "";
}

// Re-read everything that depends on the scoring model after it is changed in Settings.
async function refreshAfterModelSwitch() {
  S.meta = await api("/api/meta");
  S.summary = null;
  try {
    const s = await api("/api/summary");
    $("#nav-flagged").textContent = fmt(s.n_flagged);
    const h = await api("/api/health");
    $("#status").textContent = `${(S.meta.queue_view || {}).label || "Model"} · ${fmt(h.orders_in_queue)} orders`;
  } catch (e) { /* the page render below shows any real problem */ }
  if (typeof refreshCounts === "function") refreshCounts();
}

// ------------------------------------------------------------------ OVERVIEW
async function renderOverview() {
  const s = S.summary = await api("/api/summary");
  const r = s.risk;
  const open = s.board.to_review + s.board.investigating + s.board.customer_informed;
  const ratio = (r.bands[3].late_rate / r.bands[0].late_rate).toFixed(0);
  const brief = await api("/api/briefing").catch(() => null);
  const pats = await api("/api/patterns").catch(() => null);
  view.innerHTML = `
    <div class="hero">
      <section class="hero-main">
        <h2>Check <em>1 in ${Math.round(100 / r.flagged_share)}</em> orders. Catch <em>${r.late_orders_caught >= 52 ? "over half" : r.late_orders_caught >= 45 ? "nearly half" : r.late_orders_caught >= 30 ? "about a third to a half" : "some"}</em> of the delays.${(S.meta.queue_view || {}).in_sample ? ' <span class="tag">IN-SAMPLE</span>' : ""}</h2>
        <p>The queue holds <b>${fmt(s.n_queue)}</b> orders${(S.meta.queue_view || {}).in_sample ? "" : " the model has never seen"}, scored by <b>${esc((S.meta.queue_view || {}).label || "the serving model")}</b>. It raised an alert on <b>${fmt(s.n_flagged)}</b> of them (${r.flagged_share}%),
          and those alerts contain <b>${r.late_orders_caught}%</b> of all the orders that really arrived late.</p>
        ${viewNotice()}
        <div class="hero-actions"><a class="btn primary" href="#/queue?flagged=true">Work the alert queue</a>
          <a class="btn" href="#/briefing">Read the ${brief ? esc(brief.period_label) : "latest"} briefing</a>
          <a class="btn" href="#/patterns">See the delay patterns</a></div>
        <div style="margin-top:auto;padding-top:10px">${kicker("How often each risk band was really late")}
          <div class="bands">${r.bands.map(b => `<div class="band ${b.level}"><div class="name">${b.level}</div><div class="big">${b.late_rate}%</div>
            <div class="meta">${fmt(b.n)} orders</div></div>`).join("")}</div></div>
      </section>
      <section class="panel" style="display:flex;flex-direction:column;gap:14px">
        ${brief ? `<div class="callout"><div class="k">${esc(brief.period_label)} in one line</div>${esc(brief.story[1])}</div>` : ""}
        ${pats ? `<div><div class="k" style="font:600 11.5px var(--body);text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:6px">Biggest delay patterns</div>
          <ul class="alist">${pats.patterns.slice(0, 3).map(p => `<li><a class="pchip" href="#/patterns?p=${p.rank}">P${p.rank}</a>
            <div style="min-width:0"><div class="t">${esc(p.name)}</div><div class="d">${p.share_of_late}% of late orders · ${fmt(p.queue_alerts)} alerts in queue</div></div><span></span></li>`).join("")}</ul></div>` : ""}
      </section>
    </div>
    <div class="tiles">
      <div class="tile"><span class="k">Orders in queue</span><span class="v">${fmt(s.n_queue)}</span><span class="s">newest orders, scored at checkout</span></div>
      <div class="tile"><span class="k">Alerts raised</span><span class="v">${fmt(s.n_flagged)}</span><span class="s">risk ≥ ${pct(s.threshold, 1)} alert threshold</span></div>
      <div class="tile"><span class="k">Critical right now</span><span class="v" style="color:var(--crit)">${fmt(s.bands.CRITICAL)}</span><span class="s">risk of 80% or more</span></div>
      <div class="tile"><span class="k">Open on the board</span><span class="v">${fmt(open)}</span><span class="s">${fmt(s.board.resolved)} resolved so far</span></div>
      <div class="tile hl"><span class="k">Late orders caught</span><span class="v">${r.late_orders_caught}%</span><span class="s">by checking ${r.flagged_share}% of orders</span></div>
    </div>
    <div class="grid g-main">
      <section class="panel">
        <div class="panel-head"><div><h2>Needs attention</h2><p class="hint">Highest-risk orders not yet on the board.</p></div>
          <a class="btn" href="#/queue?band=CRITICAL">See all critical</a></div>
        <ul class="alist" id="attn">${s.needs_attention.map(o => `
          <li data-id="${esc(o.order_id)}">
            <span class="prob" style="color:${bandColor(o.risk_level)}">${pct(o.probability)}</span>
            <div style="min-width:0"><div class="t">${esc(nice(o.product_category))} · ${esc(o.seller_state)} → ${esc(o.customer_state)} <span class="mono muted">${esc(shortId(o.order_id))}</span></div>
              <div class="d">${action(o.suggested_action)}</div></div>
            <button class="btn sm" data-add="${esc(o.order_id)}">Add to board</button>
          </li>`).join("") || `<li class="muted">Every high-risk order is already on the board.</li>`}</ul>
      </section>
      <section class="panel">
        <div class="panel-head"><div><h2>Board at a glance</h2><p class="hint">Where the team's picked-up orders stand.</p></div><a class="btn" href="#/board">Open board</a></div>
        <dl class="kv" style="font-size:15px">
          <dt>To review</dt><dd>${fmt(s.board.to_review)}</dd><dt>Investigating</dt><dd>${fmt(s.board.investigating)}</dd>
          <dt>Customer informed</dt><dd>${fmt(s.board.customer_informed)}</dd><dt>Resolved</dt><dd>${fmt(s.board.resolved)}</dd>
        </dl>
        <h4 style="margin:16px 0 8px;font:600 16px var(--display);letter-spacing:.02em">Latest on the board</h4>
        <ul class="activity" id="ov-act"><li class="muted">Loading…</li></ul>
        <div class="callout" style="margin-top:16px"><div class="k">Today's main driver</div>
          ${esc(r.top_drivers[0].driver)} is behind ${Math.round(r.top_drivers[0].n / r.n_flagged * 100)}% of alerts, so most of today's risk is about peak-season capacity, not individual sellers.</div>
      </section>
    </div>
    <div class="grid g2">
      <section class="panel"><h2>Late deliveries by month</h2><p class="hint">Share of each month's orders that missed the promised date.</p>
        <div class="chart"><canvas id="c-month"></canvas></div></section>
      <section class="panel"><h2>What drives today's alerts</h2><p class="hint">Main reason behind each of the ${fmt(r.n_flagged)} alerts.</p>
        <div class="chart"><canvas id="c-drivers"></canvas></div></section>
    </div>`;
  api("/api/board/activity?limit=4").then(a => {
    $("#ov-act").innerHTML = a.items.map(x => `<li><time>${new Date(x.at).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}</time>
      <span><span class="mono muted">${esc(shortId(x.order_id))}</span> ${esc(x.detail)}</span></li>`).join("") || `<li class="muted">Nothing yet. Add an order from the list on the left.</li>`;
  }).catch(() => { });
  bindSwatches(view);
  $$("#attn li[data-id]").forEach(li => li.addEventListener("click", (e) => { if (!e.target.closest("button")) openOrder(li.dataset.id); }));
  $$("[data-add]").forEach(b => b.addEventListener("click", (e) => { e.stopPropagation(); addToBoard(b.dataset.add, b); }));
  const peak = s.monthly.reduce((a, b) => b.rate > a.rate ? b : a);
  chart($("#c-month"), { type: "line", data: { labels: s.monthly.map(m => m.month), datasets: [{ data: s.monthly.map(m => m.rate),
    borderColor: css("--accent"), backgroundColor: css("--accent-soft"), fill: true, tension: .25,
    pointRadius: s.monthly.map(m => m === peak ? 5 : 2), pointBackgroundColor: css("--accent") }] },
    options: { scales: { y: { beginAtZero: true, ticks: { callback: v => v + "%" } }, x: { grid: { display: false } } },
      plugins: { tooltip: { callbacks: { label: c => `${c.raw}% late · ${fmt(s.monthly[c.dataIndex].count)} orders` } } } } });
  hbar($("#c-drivers"), r.top_drivers.slice(0, 6).map(d => d.driver), r.top_drivers.slice(0, 6).map(d => d.n), css("--accent"), { unit: "", tip: c => `${fmt(c.raw)} alerts` });
}

// ------------------------------------------------------------------ QUEUE
async function renderQueue(params) {
  const meta = S.meta;
  const st = {
    band: params.get("band") || "", state: params.get("state") || "", category: params.get("category") || "",
    q: params.get("q") || "", flagged: params.get("flagged") || "", sort: params.get("sort") || "probability",
    order: params.get("order") || "desc", offset: +(params.get("offset") || 0),
    pattern: params.get("pattern") || "", focus: params.get("focus") || "", disagree: params.get("disagree") || "",
  };
  if (!S.patternNames) S.patternNames = (await api("/api/patterns")).patterns.map(p => ({ rank: p.rank, name: p.name }));
  const sel = new Set(st.band ? st.band.split(",") : []);
  const activeN = ["band", "state", "category", "q", "flagged", "pattern", "focus", "disagree"].filter(k => st[k]).length + (st.sort === "priority" ? 1 : 0);
  view.innerHTML = `
    <section class="panel qfilters">
      <button type="button" class="btn ghost qf-toggle" id="qf-toggle" aria-expanded="${activeN ? "true" : "false"}" aria-controls="qf-body">Filters${activeN ? ` <span class="qf-n">${activeN}</span>` : ""}<svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></button>
      <div id="qf-body">
      <div class="filters" style="justify-content:space-between">
        <div class="chips" role="group" aria-label="Risk band">
          <button class="chip any" aria-pressed="${sel.size === 0}" data-band="">All bands</button>
          ${BANDS.map(b => `<button class="chip ${b}" aria-pressed="${sel.has(b)}" data-band="${b}">${b}</button>`).join("")}
        </div>
        <div style="display:flex;gap:16px;flex-wrap:wrap">
          <label class="toggle"><input type="checkbox" id="f-flagged" ${st.flagged === "true" ? "checked" : ""}> Alerts only</label>
          <label class="toggle"><input type="checkbox" id="f-focus" ${st.focus === "true" ? "checked" : ""}> Focus categories only</label>
          ${S.meta.ensemble_available ? `<label class="toggle" title="Orders where only one of the two models raises an alert"><input type="checkbox" id="f-dis" ${st.disagree === "true" ? "checked" : ""}> Models disagree</label>` : ""}
          <label class="toggle"><input type="checkbox" id="f-priority" ${st.sort === "priority" ? "checked" : ""}> Sort by priority</label>
        </div>
      </div>
      <div class="filters" style="margin-top:12px">
        <div class="field"><label for="f-state">Customer state</label><select id="f-state"><option value="">All states</option>${meta.states.map(s => `<option ${s === st.state ? "selected" : ""}>${s}</option>`).join("")}</select></div>
        <div class="field"><label for="f-cat">Category</label><select id="f-cat"><option value="">All categories</option>${meta.categories.map(c => `<option value="${esc(c)}" ${c === st.category ? "selected" : ""}>${esc(nice(c))}</option>`).join("")}</select></div>
        <div class="field"><label for="f-pat">Delay pattern</label><select id="f-pat"><option value="">All patterns</option>${S.patternNames.map(p => `<option value="${p.rank}" ${String(p.rank) === st.pattern ? "selected" : ""}>P${p.rank} · ${esc(p.name)}</option>`).join("")}</select></div>
        <div class="field" style="flex:1;min-width:200px"><label for="f-q">Search</label><input id="f-q" value="${esc(st.q)}" placeholder="Order ID, seller ID or city"></div>
        <button class="btn ghost" id="f-reset" style="align-self:flex-end">Clear filters</button>
      </div>
      </div>
    </section>
    <section class="panel">
      <div class="table-wrap"><table class="queue-table">
        <thead><tr>
          <th>Order</th><th>Band</th><th class="num"><button data-sort="probability">Risk</button></th><th>Category</th><th>Route</th>
          <th><button data-sort="purchased_at">Placed</button></th><th class="num"><button data-sort="promised_delivery_days">Promise</button></th>
          <th>Pattern</th><th>Main driver</th><th>Suggested action</th><th></th></tr></thead>
        <tbody id="q-body"><tr><td colspan="11" class="skeleton">Loading…</td></tr></tbody>
      </table></div>
      <p class="note">Priority = risk × category weight (essential products count more; see Focus categories). Pattern = the known late-delivery pattern the order most resembles.</p>
      <div class="pager"><span id="q-count"></span><span><button class="btn sm" id="q-prev">Previous</button> <button class="btn sm" id="q-next">Next</button></span></div>
    </section>`;
  const push = (patch) => {
    const p = new URLSearchParams();
    const next = { ...st, ...patch };
    if (!("offset" in patch)) next.offset = 0;
    Object.entries(next).forEach(([k, v]) => { if (v !== "" && v != null && !(k === "offset" && +v === 0) && !(k === "sort" && v === "probability") && !(k === "order" && v === "desc")) p.set(k, v); });
    go("queue", p);
  };
  $$(".chip[data-band]").forEach(c => c.addEventListener("click", () => {
    const b = c.dataset.band;
    if (!b) return push({ band: "" });
    sel.has(b) ? sel.delete(b) : sel.add(b);
    push({ band: [...sel].join(",") });
  }));
  $("#f-flagged").addEventListener("change", e => push({ flagged: e.target.checked ? "true" : "" }));
  $("#f-state").addEventListener("change", e => push({ state: e.target.value }));
  $("#f-cat").addEventListener("change", e => push({ category: e.target.value }));
  $("#f-pat").addEventListener("change", e => push({ pattern: e.target.value }));
  if ($("#f-dis")) $("#f-dis").addEventListener("change", e => push({ disagree: e.target.checked ? "true" : "" }));
  $("#f-focus").addEventListener("change", e => push({ focus: e.target.checked ? "true" : "" }));
  $("#f-priority").addEventListener("change", e => push({ sort: e.target.checked ? "priority" : "probability", order: "desc" }));
  let qt; $("#f-q").addEventListener("input", e => { clearTimeout(qt); qt = setTimeout(() => push({ q: e.target.value.trim() }), 400); });
  $("#f-reset").addEventListener("click", () => go("queue"));
  $("#qf-toggle").addEventListener("click", e => { const o = e.currentTarget.getAttribute("aria-expanded") !== "true"; e.currentTarget.setAttribute("aria-expanded", String(o)); });
  $$("th button[data-sort]").forEach(b => b.addEventListener("click", () => push({ sort: b.dataset.sort, order: st.sort === b.dataset.sort && st.order === "desc" ? "asc" : "desc" })));

  const p = new URLSearchParams({ sort: st.sort, order: st.order, offset: st.offset, limit: 50 });
  ["band", "state", "category", "q", "flagged", "pattern", "focus", "disagree"].forEach(k => st[k] && p.set(k, st[k]));
  const res = await api("/api/orders?" + p);
  $("#q-body").innerHTML = res.items.map(o => `
    <tr class="clickable" tabindex="0" data-id="${esc(o.order_id)}">
      <td class="mono" title="${esc(o.order_id)}">${esc(shortId(o.order_id))}</td><td>${pill(o.risk_level)}</td><td class="num">${pct(o.probability, 1)}</td>
      <td>${esc(nice(o.product_category))}${o.focus_tier ? `<div><span class="tier ${esc(o.focus_tier.replace(/ /g, "-"))}">${esc(o.focus_tier)}</span></div>` : ""}</td><td style="white-space:nowrap">${esc(o.seller_state)} → ${esc(o.customer_state)}<div class="muted" style="font-size:12.5px">${fmt(o.seller_customer_distance_km)} km</div></td>
      <td style="white-space:nowrap">${date(o.purchased_at)}</td><td class="num">${fmt(o.promised_delivery_days)} d</td>
      <td><a class="pchip" href="#/patterns?p=${o.pattern}" title="${esc(patternName(o.pattern))}">P${o.pattern}</a></td>
      <td title="${esc(o.main_driver)}"><div class="clamp2">${esc(shortDriver(o.main_driver))}</div></td><td class="act" title="${esc((o.suggested_action||"").replace(/^URGENT: /,""))}"><div class="clamp2">${queueAction(o.suggested_action)}</div></td>
      <td>${o.on_board ? `<span class="tag">On board</span>` : o.flagged ? `<button class="btn sm" data-add="${esc(o.order_id)}">Add</button>` : ""}</td>
    </tr>`).join("") || `<tr><td colspan="11" class="skeleton">No orders match these filters.</td></tr>`;
  const end = Math.min(res.offset + res.limit, res.total);
  $("#q-count").textContent = res.total ? `Showing ${fmt(res.offset + 1)}–${fmt(end)} of ${fmt(res.total)} orders` : "";
  $("#q-prev").disabled = res.offset === 0;
  $("#q-next").disabled = end >= res.total;
  $("#q-prev").onclick = () => push({ offset: Math.max(0, res.offset - res.limit) });
  $("#q-next").onclick = () => push({ offset: res.offset + res.limit });
  $$("#q-body tr[data-id]").forEach(tr => tr.addEventListener("click", e => { if (!e.target.closest("button, a")) openOrder(tr.dataset.id); }));
  $$("#q-body tr[data-id]").forEach(tr => tr.addEventListener("keydown", e => { if ((e.key === "Enter" || e.key === " ") && e.target === tr) { e.preventDefault(); openOrder(tr.dataset.id); } }));
  $$("#q-body [data-add]").forEach(b => b.addEventListener("click", e => { e.stopPropagation(); addToBoard(b.dataset.add, b); }));
}

async function copyText(t) {
  try { await navigator.clipboard.writeText(t); toast("Copied"); }
  catch (e) { toast("Select the text and copy it with Ctrl+C"); }
}
function engineTag(engine, ms, fallback) {
  const label = { builtin: "Built-in", laptop: "Laptop model", gemini: "Cloud AI", auto: "Auto" }[engine] || engine;
  return `<span class="engine-tag ${fallback ? "off" : "on"}" title="${esc(fallback || "")}"><i></i>${esc(label)}${ms != null ? ` · ${(ms / 1000).toFixed(1)} s` : ""}</span>`;
}
async function rewriteInto(el, style, tagEl, btn) {
  const isField = el.tagName === "TEXTAREA";
  const text = isField ? el.value : el.innerText;
  if (btn) { btn.disabled = true; btn.textContent = "Rewriting…"; }
  try {
    const r = await api("/api/assistant/rewrite", { method: "POST", body: JSON.stringify({ text, style }) });
    if (!r.fallback_reason) {
      if (isField) el.value = r.text; else el.innerHTML = r.text.split(/\n\s*\n/).map(p => `<p>${esc(p)}</p>`).join("");
    } else toast(r.fallback_reason);
    if (tagEl) tagEl.innerHTML = engineTag(r.engine, r.elapsed_ms, r.fallback_reason);
  } catch (e) { toast(e.message); }
  if (btn) { btn.disabled = false; btn.textContent = style === "customer" ? "Polish with AI" : "Rewrite for a manager"; }
}

function patternName(rank) {
  const p = (S.patternNames || []).find(x => x.rank === rank);
  return p ? p.name : `Pattern ${rank}`;
}

// ------------------------------------------------------------------ ORDER DRAWER
const drawer = $("#drawer"), scrim = $("#scrim");
function closeDrawer() { drawer.classList.remove("open"); scrim.classList.remove("open"); drawer.setAttribute("aria-hidden", "true"); }
$("#drawer-close").addEventListener("click", closeDrawer);
scrim.addEventListener("click", closeDrawer);
document.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });

function reasonBars(reasons) {
  if (!reasons || !reasons.length) return `<p class="muted">Nothing pushes this order's risk up noticeably.</p>`;
  const max = Math.max(...reasons.map(r => Math.abs(r.contribution || 0)), 0.01);
  return `<ul class="bars">${reasons.map(r => `<li><span>${esc(r.label)}</span><span class="c">+${(r.contribution || 0).toFixed(2)}</span>
    <span class="track"><i style="width:${Math.abs(r.contribution || 0) / max * 100}%"></i></span></li>`).join("")}</ul>
    <p class="note">Bar length is each factor's push on the risk (TreeSHAP, log-odds). Only factors that raise the risk are shown.</p>`;
}
function gauge(p, lvl) {
  return `<div class="ring" style="--p:${(p * 100).toFixed(1)};--c:${bandColor(lvl)}"><span>${pct(p)}</span></div>`;
}

function handDefault(purchasedAt) {
  const d = new Date(String(purchasedAt).replace(" ", "T")); if (isNaN(d)) return "";
  d.setDate(d.getDate() + 2); const p = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

async function openOrder(id) {
  $("#drawer-title").textContent = "Order " + shortId(id);
  $("#drawer-body").innerHTML = `<div class="skeleton">Loading…</div>`;
  drawer.classList.add("open"); scrim.classList.add("open"); drawer.setAttribute("aria-hidden", "false");
  let o;
  try { o = await api("/api/orders/" + encodeURIComponent(id)); }
  catch (e) { $("#drawer-body").innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
  const f = o.features, sh = o.seller_history, out = o.outcome;
  $("#drawer-body").innerHTML = `
    <div class="gauge">${gauge(o.probability, o.risk_level)}
      <div>${pill(o.risk_level)} ${o.flagged ? `<span class="tag">Alert</span>` : ""}
        <div style="font:600 20px var(--display);margin-top:6px">${esc(nice(o.product_category))}</div>
        <div class="muted">${esc(o.seller_city || o.seller_state)}, ${esc(o.seller_state)} → ${esc(o.customer_city)}, ${esc(o.customer_state)}</div>
        <div class="mono muted" style="margin-top:4px">${esc(o.order_id)}</div></div></div>
    <div class="callout"><div class="k">Suggested action</div>${action(o.suggested_action)}</div>
    <section class="handoff" id="d-hand"><div class="k">Carrier has collected it?</div>
      <p class="muted" style="margin:2px 0 8px;font-size:13.5px">Enter when the carrier picked the parcel up to get a second, sharper score.</p>
      <div class="hand-row"><label class="sr" for="d-hand-at">Handed to carrier at</label>
        <input type="datetime-local" id="d-hand-at" value="${handDefault(o.purchased_at)}"><button class="btn sm primary" id="d-hand-go" type="button">Re-score</button></div>
      <div id="d-hand-out" aria-live="polite"></div></section>
    <section class="handoff" id="d-trk"><div class="k">In transit? Tracking score </div>
      <p class="muted" style="margin:2px 0 8px;font-size:13.5px">Re-score using the carrier scans, failed attempts, weather and strikes known <i>N days after the order was placed</i>. The tracking data is invented, because Olist has none.</p>
      <div class="hand-row"><label class="sr" for="d-trk-days">Days after purchase</label>
        <input type="number" id="d-trk-days" min="0" max="120" step="1" value="7"><button class="btn sm primary" id="d-trk-go" type="button">Check tracking</button></div>
      <div id="d-trk-out" aria-live="polite"></div></section>
    <div class="chips2"><a class="pchip" href="#/patterns?p=${o.pattern}" data-close>P${o.pattern}</a><span class="muted" style="font-size:13.5px">Resembles: ${esc(o.pattern_name)}</span>
      ${o.focus_tier ? `<span class="tier ${esc(o.focus_tier.replace(/ /g, "-"))}">${esc(o.focus_tier)}</span>` : ""}</div>
    <section><h4 style="margin:0 0 10px;font:600 17px var(--display);letter-spacing:.02em">The story</h4><div class="story" id="d-story"><p class="muted">Writing…</p></div>
      <div class="story-tools"><button class="btn sm" id="d-copy">Copy</button><button class="btn sm" id="d-rewrite">Rewrite for a manager</button><span id="d-story-eng"></span></div></section>
    <section><h4 style="margin:0 0 10px;font:600 17px var(--display);letter-spacing:.02em">Why this score</h4>${reasonBars(o.reasons)}</section>
    ${o.calibrated_probability != null ? `<section class="callout"><div class="k">Estimated real chance of being late</div><b style="font:700 26px var(--display)">${pct(o.calibrated_probability, 0)}</b>
      <span class="muted"> · the score above ranks orders; this is how often orders with a score like this were actually late (the typical order: ${pct(S.meta.base_rate || 0.081, 0)}).</span></section>` : ""}
    ${o.second_opinion ? `<section class="callout ${o.second_opinion.agrees ? "" : "warn"}"><div class="k">Second opinion · ${esc(o.second_opinion.label)}</div>
      ${pill(o.second_opinion.risk_level)} ${pct(o.second_opinion.probability, 1)} risk, ${o.second_opinion.flagged ? "would raise an alert" : "would not raise an alert"}.
      ${o.second_opinion.agrees ? "Both models agree on the alert decision." : "The two models disagree on the alert decision, so this order is worth a closer look."}
      <p class="muted" style="margin:6px 0 0;font-size:12.5px">The older model was trained on most of these orders, so its score is a comparison, not an independent check.</p></section>` : ""}
    <details class="fold"><summary>Order facts at checkout</summary>
      <dl class="kv">
        <dt>Placed</dt><dd>${date(o.purchased_at)}</dd>
        <dt>Promised delivery</dt><dd>${date(o.promised_by)} (${fmt(f.promised_delivery_days, 1)} days)</dd>
        <dt>Distance</dt><dd>${fmt(f.seller_customer_distance_km)} km</dd>
        <dt>Order value</dt><dd>R$ ${fmt(f.total_order_value, 2)}</dd>
        <dt>Items / sellers</dt><dd>${fmt(f.n_items)} / ${fmt(f.n_distinct_sellers)}</dd>
        <dt>Weight</dt><dd>${fmt(f.total_weight_g / 1000, 2)} kg</dd>
        <dt>Payment</dt><dd>${esc(nice(f.payment_type))}, ${fmt(f.n_payment_installments)} instalment(s)</dd>
        <dt>Approval delay</dt><dd>${fmt(f.approval_delay_hours, 1)} h</dd>
        <dt>Seller's past late rate</dt><dd>${pct(f.seller_historical_delay_rate, 1)}</dd>
        <dt>Region's past late rate</dt><dd>${pct(f.customer_state_historical_delay_rate, 1)}</dd>
      </dl></details>
    ${sh ? `<section class="callout outcome"><div class="k">Seller record (full history)</div>
      ${fmt(sh.n_orders)} delivered orders, ${pct(sh.rate, 1)} late. <a href="#/queue?q=${esc(o.seller_id)}" data-close>See this seller's orders</a></section>` : ""}
    <details class="fold"><summary>Logistics view</summary>
      <dl class="kv"><dt>Delivery model</dt><dd>${esc(o.simulated.delivery_model)}</dd><dt>Hub transfers</dt><dd>${o.simulated.hub_transfers}</dd>
        <dt>Shipping mode</dt><dd>${esc(o.simulated.shipping_mode)}</dd><dt>Carrier</dt><dd>${esc(o.simulated.carrier)}</dd>
        <dt>Route</dt><dd>${esc(o.simulated.first_mile_hub)} → ${esc(o.simulated.last_mile_hub)} · ~${fmt(o.simulated.route_km)} km</dd></dl>
      <p class="note">Olist doesn't publish how orders moved, so these routing fields are derived from the route, promise and parcel. They are not used by the risk model.</p></details>
    <details class="fold"><summary>Message to the customer</summary>
      <textarea id="d-msg" rows="8" style="width:100%;border:1px solid var(--line);border-radius:8px;padding:10px;background:var(--panel)" aria-label="Draft customer message"></textarea>
      <div class="story-tools"><button class="btn sm" id="d-msg-copy">Copy message</button><button class="btn sm" id="d-msg-polish">Polish with AI</button><span id="d-msg-eng"></span></div></details>
    <section class="callout outcome"><div class="k">What actually happened · not used by the model</div>
      <span class="${out.was_late ? "late" : "ontime"}">${out.was_late ? "Arrived late" : "Arrived on time"}</span>
      ${out.days_vs_promise != null ? ` · ${Math.abs(out.days_vs_promise).toFixed(1)} days ${out.days_vs_promise > 0 ? "after" : "before"} the promised date` : ""}</section>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      ${o.board ? `<a class="btn" href="#/board" data-close>On the board · ${esc(o.board.stage.replace(/_/g, " "))}</a>` : `<button class="btn primary" id="d-add">Add to action board</button>`}
      <a class="btn" href="#/simulate?from=${esc(o.order_id)}" data-close>Try changes in the simulator</a>
    </div>`;
  $$("#drawer-body [data-close]").forEach(a => a.addEventListener("click", closeDrawer));
  api(`/api/orders/${encodeURIComponent(o.order_id)}/story`).then(st => {
    $("#d-story").innerHTML = st.paragraphs.map(p => `<p>${esc(p)}</p>`).join("");
    $("#d-msg").value = st.customer_message;
  }).catch(e => { $("#d-story").innerHTML = `<p class="err">${esc(e.message)}</p>`; });
  $("#d-copy").addEventListener("click", () => copyText($("#d-story").innerText));
  $("#d-rewrite").addEventListener("click", () => rewriteInto($("#d-story"), "executive", $("#d-story-eng"), $("#d-rewrite")));
  $("#d-msg-copy").addEventListener("click", () => copyText($("#d-msg").value));
  $("#d-msg-polish").addEventListener("click", () => rewriteInto($("#d-msg"), "customer", $("#d-msg-eng"), $("#d-msg-polish")));
  const add = $("#d-add");
  if (add) add.addEventListener("click", async () => { await addToBoard(o.order_id, add); });
  $("#d-trk-go").addEventListener("click", async () => {
    const days = $("#d-trk-days").value, out = $("#d-trk-out"), btn = $("#d-trk-go");
    if (days === "" || isNaN(+days)) { out.innerHTML = `<p class="err">Enter the number of days.</p>`; return; }
    btn.disabled = true; out.innerHTML = `<p class="muted">Scoring…</p>`;
    try {
      const [r, t] = await Promise.all([
        api("/api/score/tracking", { method: "POST", body: JSON.stringify({ order_id: o.order_id, days_after_purchase: +days }) }),
        api(`/api/tracking/${encodeURIComponent(o.order_id)}?days_after_purchase=${encodeURIComponent(days)}`)]);
      const up = r.change > 0.005, down = r.change < -0.005;
      const ev = t.events.map(x => `<li><span class="mono">${esc(x.time.slice(0, 16))}</span> · ${esc(nice(x.type.toLowerCase()))}${x.state ? " (" + esc(x.state) + ")" : ""}</li>`).join("");
      const c = t.context, wl = ["none", "rain", "heavy rain", "storm"];
      out.innerHTML = `<div class="hand-res">${pill(r.risk_level)} <b class="prob">${pct(r.probability)}</b> ${r.flagged ? `<span class="tag">Alert</span>` : ""}
        <div class="muted" style="font-size:13.5px;margin-top:4px">${up ? "Up" : down ? "Down" : "Unchanged"} from ${pct(r.checkout_probability)} at checkout · day ${fmt(r.days_since_purchase, 1)} · ${fmt(r.days_left, 1)} days of the promise left · ${r.scans} hub scans · ${r.failed_attempts} failed attempt(s)${r.longest_silence_hours != null ? " · longest silence " + fmt(r.longest_silence_hours, 0) + " h" : ""}</div>
        <div class="muted" style="font-size:13.5px">Destination weather (3 days): ${wl[Math.round(c.dest_weather_3d_max || 0)]}${c.dest_strike_now ? " · strike active at the destination" : ""}${c.origin_strike_now ? " · strike active at the seller's state" : ""}</div>
        ${r.reasons.length ? `<ul class="hand-why">${r.reasons.map(x => `<li>${esc(x.label)}</li>`).join("")}</ul>` : ""}
        ${ev ? `<details class="fold"><summary>Scans so far (${t.events.length})</summary><ul class="hand-why">${ev}</ul></details>` : `<p class="muted" style="font-size:13.5px">No scans yet at that time.</p>`}</div>`;
    } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
    btn.disabled = false;
  });
  $("#d-hand-go").addEventListener("click", async () => {
    const at = $("#d-hand-at").value, out = $("#d-hand-out"), btn = $("#d-hand-go");
    if (!at) { out.innerHTML = `<p class="err">Pick the date and time first.</p>`; return; }
    btn.disabled = true; out.innerHTML = `<p class="muted">Scoring…</p>`;
    try {
      const r = await api("/api/score/handoff", { method: "POST", body: JSON.stringify({ order_id: o.order_id, handed_over_at: at }) });
      const up = r.change > 0.005, down = r.change < -0.005;
      out.innerHTML = `<div class="hand-res">${pill(r.risk_level)} <b class="prob">${pct(r.probability)}</b> ${r.flagged ? `<span class="tag">Alert</span>` : ""}
        <div class="muted" style="font-size:13.5px;margin-top:4px">${up ? "Up" : down ? "Down" : "Unchanged"} from ${pct(r.checkout_probability)} at checkout ·
          ${fmt(r.days_left, 1)} days of the promise left · seller took ${fmt(r.handling_hours, 0)} h to hand over</div>
        ${r.reasons.length ? `<ul class="hand-why">${r.reasons.map(x => `<li>${esc(x.label)}</li>`).join("")}</ul>` : ""}</div>`;
    } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
    btn.disabled = false;
  });
}

// ------------------------------------------------------------------ BOARD
function ageText(iso) {
  const h = Math.max(0, (Date.now() - new Date(iso).getTime()) / 36e5);
  return h < 1 ? "just now" : h < 48 ? Math.round(h) + " h" : Math.round(h / 24) + " d";
}
async function renderBoard() {
  const b = await api("/api/board");
  const act = await api("/api/board/activity?limit=8");
  const idx = Object.fromEntries(b.stages.map((s, i) => [s.key, i]));
  const byKey = Object.fromEntries(b.stages.map(s => [s.key, s]));
  const owners = [...new Set(b.items.map(i => i.owner).filter(Boolean))];
  const filt = renderBoard.filter || { owner: "", critical: false };
  renderBoard.filter = filt;
  if (filt.owner && !owners.includes(filt.owner)) filt.owner = "";   // that person has no orders left: show everyone
  const shown = b.items.filter(it => (!filt.owner || it.owner === filt.owner) && (!filt.critical || it.order.risk_level === "CRITICAL"));

  const card = (it) => {
    const o = it.order, st = byKey[it.stage], i = idx[it.stage];
    const next = st.next_label && i < b.stages.length - 1 ? `<button class="btn sm primary" data-move="1" title="${esc(st.do)}">${esc(st.next_label)} →</button>` : `<button class="btn sm" data-move="reopen">Reopen</button>`;
    return `<article class="card" draggable="true" data-id="${esc(it.order_id)}" aria-label="Order ${esc(shortId(it.order_id))}">
      <div class="dots" aria-label="Stage ${i + 1} of ${b.stages.length}">${b.stages.map((_, k) => `<i class="${k <= i ? "on" : ""}"></i>`).join("")}</div>
      <div class="top">${pill(o.risk_level)}<span class="id">${esc(shortId(it.order_id))}</span><b class="prob">${pct(o.probability)}</b></div>
      <div class="what"><b>${esc(nice(o.product_category))}</b> · ${esc(o.seller_state)} → ${esc(o.customer_state)}</div>
      <div class="meta"><span>Due ${date(o.promised_by)}</span>${it.owner ? `<span class="tag">${esc(it.owner)}</span>` : `<span class="tag warn">No owner</span>`}</div>
      ${it.stage !== "resolved" && o.suggested_action ? `<div class="sugg"><b>Suggested:</b> ${esc(o.suggested_action)}</div>` : ""}
      ${it.note ? `<div class="note-text clamp">${esc(it.note)}</div>` : ""}
      <div class="meta"><span class="muted" title="Last change: ${esc(new Date(it.updated_at).toLocaleString("en-IN"))}">${ageText(it.created_at)} on board · updated ${ageText(it.updated_at) === "just now" ? "just now" : ageText(it.updated_at) + " ago"}</span>${it.stage === "customer_informed" ? `<button class="btn sm ghost" data-copy>Copy message</button>` : ""}</div>
      ${it.stage === "resolved" && it.outcome ? `<div class="meta">Outcome: <span class="${it.outcome.was_late ? "late" : "ontime"}">${it.outcome.was_late ? "arrived late" : "arrived on time"}</span></div>` : ""}
      <div class="actions">${next}<span class="sp"></span>
        <details class="more"><summary class="btn sm ghost" aria-label="More actions">⋯</summary>
          <div class="menu">
            <button data-edit>Owner and note</button>
            <button data-open>Order details</button>
            ${i > 0 ? `<button data-move="-1">Move back to ${esc(b.stages[i - 1].label)}</button>` : ""}
            <button data-remove class="danger">Remove from board</button>
          </div></details>
      </div></article>`;
  };

  view.innerHTML = `
    <div class="board-top">
      <div class="bsteps" role="list">${b.stages.map((s, n) => `<div class="bstep" role="listitem"><span class="num">${n + 1}</span><div><b>${esc(s.label)}</b><small>${esc(s.meaning)}</small></div></div>`).join("")}</div>
      <div class="board-tools">
        <button class="btn primary" id="autofill" title="Adds the 10 riskiest critical orders that are not on the board yet">Add top 10 critical</button>
        <a class="btn" href="#/queue?flagged=true">Pick from queue</a>
        <label class="sr" for="f-owner">Filter by owner</label>
        <select id="f-owner"><option value="">Everyone</option>${owners.map(o => `<option ${o === filt.owner ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>
        <label class="chk"><input type="checkbox" id="f-crit" ${filt.critical ? "checked" : ""}> Critical only</label>
      </div>
    </div>
    <details class="panel guide"><summary><b>What does each stage do?</b> <span class="muted">Click to read the workflow</span></summary>
      <div class="guide-grid">${b.stages.map((s, n) => `<div><h3>${n + 1}. ${esc(s.label)}</h3><p>${esc(s.meaning)}</p><ul>${(s.steps || []).map(x => `<li>${esc(x)}</li>`).join("")}</ul><p class="muted"><b>Move on when:</b> ${esc(s.done_when || "")}</p></div>`).join("")}</div>
      <div class="table-wrap"><table class="cmp"><thead><tr><th>Stage</th><th>Who is acting</th><th>Customer told?</th><th>Typical time</th><th>The difference</th></tr></thead><tbody>${b.stages.map(s => `<tr><td><b>${esc(s.label)}</b></td><td>${esc(s.who || "")}</td><td>${esc(s.customer_told || "")}</td><td>${esc(s.typical_time || "")}</td><td>${esc(s.do)}</td></tr>`).join("")}</tbody></table></div>
      <p class="note"><b>To review → Investigating:</b> someone has taken ownership and started checking. <b>Investigating → Customer informed:</b> the facts are confirmed and the customer has been told. <b>Customer informed → Resolved:</b> the parcel arrived or the case is settled.</p>
      <p class="note">Drag a card between columns, or use its button. A move only records where the case stands: OnTime does not contact the seller, carrier or customer, so write what you actually did in the note. Every move is logged in Recent activity.</p></details>
    ${b.items.length === 0 ? `<div class="callout"><div class="k">The board is empty</div>Press <b>Add top 10 critical</b> to start with the riskiest orders.</div>` : ""}
    <div class="board" id="board">${b.stages.map(s => {
      const items = shown.filter(it => it.stage === s.key);
      return `<section class="col" data-stage="${s.key}"><header><div><b>${esc(s.label)}</b><small>${esc(s.do)}</small></div><span class="n">${items.length}</span></header>
        <div class="cards">${items.map(card).join("") || `<div class="empty">Nothing here</div>`}</div></section>`;
    }).join("")}</div>
    <details class="panel recent"><summary><b>Recent activity</b></summary>
      <ul class="activity">${act.items.map(a => `<li><time>${new Date(a.at).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}</time>
        <span><span class="mono muted">${esc(shortId(a.order_id))}</span> ${esc(a.detail)}</span></li>`).join("") || `<li class="muted">No activity yet.</li>`}</ul></details>`;

  const move = async (id, stage) => {
    const cur = b.items.find(x => x.order_id === id);
    if (!cur || !byKey[stage] || cur.stage === stage) return;   // dropped back on its own column: nothing to record
    if ((stage === "customer_informed" || stage === "resolved") && !(cur.note || "").trim()
        && !confirm("This order has no note yet. A move only records status, so the note is the only record of what was done (who was contacted, the confirmed ETA). Move it anyway?")) return;
    try { await api("/api/board/" + id, { method: "PATCH", body: JSON.stringify({ stage }) }); toast("Status recorded: " + byKey[stage].label + ". " + (byKey[stage].do || "")); await renderBoard(); refreshCounts(); }
    catch (e) { toast(e.message); }
  };
  $("#autofill").addEventListener("click", async () => {
    try {
      const r = await api("/api/board/autofill?n=10", { method: "POST" });
      toast(r.added ? `Added ${r.added} critical orders` : "All critical orders are already on the board");
      await renderBoard(); refreshCounts();
    } catch (e) { toast(e.message); }
  });
  $("#f-owner").addEventListener("change", e => { filt.owner = e.target.value; renderBoard(); });
  $("#f-crit").addEventListener("change", e => { filt.critical = e.target.checked; renderBoard(); });

  $$(".card").forEach(c => {
    const id = c.dataset.id, it = b.items.find(x => x.order_id === id), i = idx[it.stage];
    c.addEventListener("dragstart", e => { e.dataTransfer.setData("text/plain", id); c.classList.add("dragging"); });
    c.addEventListener("dragend", () => c.classList.remove("dragging"));
    $$("[data-move]", c).forEach(btn => btn.addEventListener("click", () => {
      const d = btn.dataset.move;
      move(id, d === "reopen" ? b.stages[0].key : b.stages[i + +d].key);
    }));
    $("[data-open]", c).addEventListener("click", () => openOrder(id));
    const cp = $("[data-copy]", c);
    if (cp) cp.addEventListener("click", async () => {
      const msg = `Hello, a quick update on your order ${shortId(id)}: it may reach you after the date we first promised (${date(it.order.promised_by)}). We are in touch with the seller and carrier and will confirm a new delivery date shortly. We apologise for the inconvenience.`;
      try { await navigator.clipboard.writeText(msg); toast("Message copied. OnTime does not send it: send it yourself, then record it in the note."); } catch { toast("Copy failed; select the text manually"); }
    });
    $("[data-remove]", c).addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      if (btn.dataset.confirm !== "1") { btn.dataset.confirm = "1"; btn.textContent = "Click again to confirm"; return; }
      try { await api("/api/board/" + id, { method: "DELETE" }); toast("Removed from the board"); await renderBoard(); refreshCounts(); }
      catch (err) { toast(err.message); }
    });
    $("[data-edit]", c).addEventListener("click", () => {
      $("details.more", c).open = false;
      if ($(".edit", c)) return;
      const box = document.createElement("div"); box.className = "edit";
      box.innerHTML = `<div class="field"><label for="o-${id}">Owner</label><input id="o-${id}" value="${esc(it.owner)}" maxlength="80" placeholder="Who is handling this?"></div>
        <div class="field" style="margin-top:6px"><label for="n-${id}">Note</label><textarea id="n-${id}" rows="3" maxlength="2000" placeholder="What was done, who was contacted">${esc(it.note)}</textarea></div>
        <div style="display:flex;gap:6px;margin-top:8px"><button class="btn sm primary" data-save>Save</button><button class="btn sm ghost" data-cancel>Cancel</button></div>`;
      c.insertBefore(box, $(".actions", c));
      c.draggable = false;
      $("[data-cancel]", box).addEventListener("click", () => { box.remove(); c.draggable = true; });
      $("[data-save]", box).addEventListener("click", async () => {
        try { await api("/api/board/" + id, { method: "PATCH", body: JSON.stringify({ owner: $(`#o-${id}`).value, note: $(`#n-${id}`).value }) }); toast("Saved"); renderBoard(); }
        catch (err) { toast(err.message); }
      });
    });
  });
  $$(".col").forEach(col => {
    col.addEventListener("dragover", e => { e.preventDefault(); col.classList.add("drop"); });
    col.addEventListener("dragleave", () => col.classList.remove("drop"));
    col.addEventListener("drop", e => { e.preventDefault(); col.classList.remove("drop"); const id = e.dataTransfer.getData("text/plain"); if (id) move(id, col.dataset.stage); });
  });
}

// ------------------------------------------------------------------ SIMULATOR
const SIM_FIELDS = [
  { set: "Route and seller", fields: [
    { k: "seller_id", label: "Seller ID", type: "text", help: "Optional. Looks up the seller's late-delivery record." },
    { k: "seller_state", label: "Seller state", type: "state" },
    { k: "customer_state", label: "Customer state", type: "state" },
    { k: "seller_customer_distance_km", label: "Distance (km)", type: "range", min: 0, max: 3500, step: 10 },
  ] },
  { set: "Promise and timing", fields: [
    { k: "promised_delivery_days", label: "Promised delivery (days)", type: "range", min: 1, max: 60, step: 1 },
    { k: "purchase_month", label: "Month", type: "month" },
    { k: "purchase_dow", label: "Day of week", type: "dow" },
    { k: "purchase_hour", label: "Hour of day", type: "number", min: 0, max: 23, step: 1 },
    { k: "approval_delay_hours", label: "Payment approval delay (h)", type: "number", min: 0, step: 0.5 },
  ] },
  { set: "Basket", fields: [
    { k: "product_category", label: "Category", type: "category" },
    { k: "n_items", label: "Items", type: "number", min: 1, max: 20, step: 1 },
    { k: "n_distinct_sellers", label: "Sellers in order", type: "number", min: 1, max: 5, step: 1 },
    { k: "total_price", label: "Price (R$)", type: "number", min: 0, step: 1 },
    { k: "total_freight", label: "Freight (R$)", type: "number", min: 0, step: 0.5 },
    { k: "total_weight_g", label: "Weight (g)", type: "number", min: 0, step: 50 },
  ] },
  { set: "Payment", fields: [
    { k: "payment_type", label: "Payment type", type: "payment" },
    { k: "n_payment_installments", label: "Instalments", type: "number", min: 1, max: 24, step: 1 },
  ] },
  { set: "Dispatch checkpoint inputs (for operational model)", fields: [
    { k: "weather_severity_forecast", label: "Forecast weather severity (0–5)", type: "number", min: 0, max: 5, step: 0.1, help: "Use a current forecast if available; 0 is calm, 5 is severe." },
    { k: "congestion_forecast", label: "Forecast congestion (0–5)", type: "number", min: 0, max: 5, step: 0.1 },
    { k: "carrier_reliability_prior", label: "Carrier reliability prior (0–1)", type: "number", min: 0, max: 1, step: 0.01 },
    { k: "capacity_utilization_at_dispatch", label: "Capacity utilisation", type: "number", min: 0.3, max: 1.3, step: 0.01 },
    { k: "remaining_sla_slack_hours", label: "Remaining SLA slack (hours)", type: "number", min: -10, max: 1000, step: 1 },
    { k: "facility_backlog_at_dispatch", label: "Facility backlog (orders)", type: "number", min: 0, step: 1 },
    { k: "route_disruption_alert_at_dispatch", label: "Route disruption alert (0/1)", type: "number", min: 0, max: 1, step: 1 },
    { k: "last_scan_age_hours", label: "Age of last scan (hours)", type: "number", min: 0, step: 0.5 },
    { k: "estimated_truck_delay_hours_at_dispatch", label: "Estimated truck delay (hours)", type: "number", min: 0, step: 0.5 },
    { k: "dispatch_lag_hours", label: "Purchase-to-dispatch lag (hours)", type: "number", min: 0.25, max: 96, step: 0.25 },
  ] },
];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DOWS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function fieldHtml(f, v) {
  const id = "s-" + f.k;
  const opt = (vals, labels) => vals.map((x, i) => `<option value="${esc(x)}" ${String(x) === String(v) ? "selected" : ""}>${esc(labels ? labels[i] : nice(x))}</option>`).join("");
  let input;
  if (f.type === "state") input = `<select id="${id}">${opt(S.meta.states)}</select>`;
  else if (f.type === "category") input = `<select id="${id}">${opt(S.meta.categories)}</select>`;
  else if (f.type === "payment") input = `<select id="${id}">${opt(S.meta.payment_types)}</select>`;
  else if (f.type === "month") input = `<select id="${id}">${opt(MONTHS.map((_, i) => i + 1), MONTHS)}</select>`;
  else if (f.type === "dow") input = `<select id="${id}">${opt(DOWS.map((_, i) => i), DOWS)}</select>`;
  else if (f.type === "range") input = `<div class="range-row"><input type="range" id="${id}" min="${f.min}" max="${f.max}" step="${f.step}" value="${esc(v)}"><output id="${id}-o">${fmt(v)}</output></div>`;
  else if (f.type === "text") input = `<input id="${id}" value="${esc(v ?? "")}" placeholder="e.g. 7c67e144…" autocomplete="off">`;
  else input = `<input type="number" id="${id}" value="${esc(v)}" ${f.min != null ? `min="${f.min}"` : ""} ${f.max != null ? `max="${f.max}"` : ""} step="${f.step || "any"}">`;
  return `<div class="field" ${f.type === "range" || f.type === "text" ? 'style="grid-column:span 2"' : ""}><label for="${id}">${esc(f.label)}</label>${input}${f.help ? `<span class="help">${esc(f.help)}</span>` : ""}</div>`;
}

const PRESETS = {
  typical: { label: "Typical order", patch: {} },
  northeast: { label: "Long route to the North-East", patch: { seller_state: "SP", customer_state: "AL", seller_customer_distance_km: 2100, promised_delivery_days: 25 } },
  peak: { label: "Tight promise in peak season", patch: { purchase_month: 3, promised_delivery_days: 7, seller_customer_distance_km: 700, customer_state: "RJ" } },
  local: { label: "Same-city, relaxed promise", patch: { seller_state: "SP", customer_state: "SP", seller_customer_distance_km: 15, promised_delivery_days: 20, purchase_month: 6 } },
};

async function renderSimulate(params) {
  const t = S.meta.typical_order;
  const base = {};
  SIM_FIELDS.forEach(g => g.fields.forEach(f => { base[f.k] = t[f.k] ?? ""; }));
  base.seller_id = "";
  base.purchase_month = Math.round(base.purchase_month); base.purchase_dow = Math.round(base.purchase_dow); base.purchase_hour = Math.round(base.purchase_hour);
  Object.assign(base, { weather_severity_forecast: "", congestion_forecast: "", carrier_reliability_prior: "",
    capacity_utilization_at_dispatch: "", remaining_sla_slack_hours: "", facility_backlog_at_dispatch: "",
    route_disruption_alert_at_dispatch: "", last_scan_age_hours: "", estimated_truck_delay_hours_at_dispatch: "", dispatch_lag_hours: "" });
  let from = null;
  if (params.get("from")) {
    try {
      from = await api("/api/orders/" + encodeURIComponent(params.get("from")));
      SIM_FIELDS.forEach(g => g.fields.forEach(f => { if (f.k in from.features) base[f.k] = from.features[f.k]; }));
      base.seller_id = from.seller_id || "";
      ["seller_customer_distance_km", "promised_delivery_days"].forEach(k => base[k] = Math.round(base[k]));
    } catch (e) { toast(e.message); }
  }
  view.innerHTML = `
    <p class="lede">Every number here comes from the same saved model the queue uses. Change a field and the score updates. Fields not on the form come from a typical order, and the result says which.</p>
    ${(S.meta.ensemble_available || S.meta.enriched_available || S.meta.synthetic_operational_available) ? `<div class="field" style="max-width:300px;margin-bottom:10px"><label for="sim-model">Scoring model</label><select id="sim-model"><option value="">As set in Settings</option><option value="xgboost">Tuned XGBoost</option>${S.meta.ensemble_available ? `<option value="ensemble">Weighted ensemble</option>` : ""}${S.meta.enriched_available ? `<option value="enriched">Enriched chronological model</option>` : ""}${S.meta.synthetic_operational_available ? `<option value="synthetic_operational">Dispatch operational model</option>` : ""}</select><p class="note">The dispatch model requires dispatch-time values. Missing values are estimated and identified in the result.</p></div>` : ""}
    <div class="chips" id="presets">${Object.entries(PRESETS).map(([k, p]) => `<button class="chip" data-preset="${k}">${esc(p.label)}</button>`).join("")}
      <button class="chip" id="load-critical">Load a critical order</button></div>
    ${from ? `<div class="callout outcome"><div class="k">Loaded from the queue</div>Order <span class="mono">${esc(shortId(from.order_id))}</span> (${pct(from.probability)} in the queue). It ${from.outcome.was_late ? "arrived late" : "arrived on time"}.</div>` : ""}
    <div class="grid g-main">
      <section class="panel"><form id="sim-form" autocomplete="off">${SIM_FIELDS.map(g => `<fieldset class="fieldset"><legend>${esc(g.set)}</legend>
        <div class="form-grid">${g.fields.map(f => fieldHtml(f, base[f.k])).join("")}</div></fieldset>`).join("")}</form></section>
      <section class="panel result" id="sim-result"><div class="skeleton">Scoring…</div></section>
    </div>`;
  let baseline = null, timer = null, seq = 0;
  const read = () => {
    const o = {};
    SIM_FIELDS.forEach(g => g.fields.forEach(f => {
      const el = $("#s-" + f.k); let v = el.value;
      if (["number", "range", "month", "dow"].includes(f.type)) v = v === "" ? null : Number(v);
      o[f.k] = v;
    }));
    if (!o.seller_id) delete o.seller_id;
    return o;
  };
  const setValues = (patch) => {
    Object.entries({ ...base, ...patch }).forEach(([k, v]) => { const el = $("#s-" + k); if (el) { el.value = v; const out = $(`#s-${k}-o`); if (out) out.textContent = fmt(v); } });
  };
  const run = async () => {
    const my = ++seq;
    let r;
    try { r = await api("/api/score", { method: "POST", body: JSON.stringify({ fields: read(), model: ($("#sim-model") || {}).value || null }) }); }
    catch (e) { $("#sim-result").innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
    if (my !== seq) return;
    if (!baseline && r.status === "SCORED") baseline = r.probability;
    const box = $("#sim-result");
    if (r.status !== "SCORED") {
      box.innerHTML = `<h2>Not scored</h2><div class="callout bad"><div class="k">Rejected by the guardrails</div><ul style="margin:4px 0 0 18px;padding:0">${r.reject_reasons.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>
        <p class="note">The system refuses to turn impossible input into a confident-looking number.</p>`;
      return;
    }
    const delta = baseline != null ? r.probability - baseline : 0;
    const filledLabels = r.filled_from_typical.map(k => S.meta.feature_labels[k] || nice(k));
    box.innerHTML = `
      <div class="gauge">${gauge(r.probability, r.risk_level)}
        <div>${pill(r.risk_level)} ${r.flagged ? `<span class="tag">Would raise an alert</span>` : `<span class="tag">No alert</span>`}
          <div class="muted" style="margin-top:6px">Risk ${pct(r.probability, 1)} · alert threshold ${pct(r.threshold, 1)} · ${esc(r.model_label || "")}</div>${r.calibrated_probability != null ? `<div class="muted" style="margin-top:4px">Estimated real chance of being late: <b>${pct(r.calibrated_probability, 0)}</b></div>` : ""}
          ${!r.flagged && r.risk_level === "HIGH" ? `<div class="muted" style="font-size:13px">In the HIGH band (60%+) but just under the alert line.</div>` : ""}
          ${Math.abs(delta) >= 0.005 ? `<div style="margin-top:4px;font-weight:600;color:${delta > 0 ? "var(--crit)" : "var(--low)"}">${delta > 0 ? "▲" : "▼"} ${(Math.abs(delta) * 100).toFixed(1)} pts vs. where you started</div>` : ""}
        </div></div>
      <div class="callout" style="margin-top:14px"><div class="k">Suggested action</div>${action(r.suggested_action)}</div>
      <h4 style="margin:16px 0 10px;font:600 17px var(--display);letter-spacing:.02em">What pushes the risk up</h4>${reasonBars(r.reasons)}
      <dl class="kv" style="margin-top:14px"><dt>Seller's past late rate used</dt><dd>${pct(r.rates_used.seller_historical_delay_rate, 1)}</dd>
        <dt>Category's past late rate</dt><dd>${pct(r.rates_used.category_historical_delay_rate, 1)}</dd>
        <dt>Region's past late rate</dt><dd>${pct(r.rates_used.customer_state_historical_delay_rate, 1)}</dd></dl>
      ${r.pattern ? `<div class="callout outcome" style="margin-top:14px"><div class="k">Resembles delay pattern P${r.pattern.rank}</div>
        <a href="#/patterns?p=${r.pattern.rank}">${esc(r.pattern.name)}</a>. ${esc(r.pattern.recommendation)}</div>` : ""}
      ${r.simulated ? `<p class="note">Logistics: ${esc(r.simulated.delivery_model)}, ${r.simulated.hub_transfers} hub transfer(s), ${esc(r.simulated.shipping_mode)} via ${esc(r.simulated.carrier)}.</p>` : ""}
      ${r.warnings.length ? `<div class="callout warn" style="margin-top:14px"><div class="k">Low confidence</div>${r.warnings.map(esc).join("<br>")}</div>` : ""}
      ${r.notes.length ? `<p class="note">${r.notes.map(esc).join(" ")}</p>` : ""}
      <p class="note">Taken from a typical order: ${esc(filledLabels.join(", ") || "nothing")}.</p>
      ${provNote(r.provenance)}
      <p class="note">The drivers above are model associations, not confirmed causes of delay.</p>`;
  };
  const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 250); };
  $("#sim-form").addEventListener("input", e => { const out = $(`#${e.target.id}-o`); if (out) out.textContent = fmt(e.target.value); schedule(); });
  $("#sim-form").addEventListener("submit", e => e.preventDefault());
  if ($("#sim-model")) $("#sim-model").addEventListener("change", () => { baseline = null; run(); });
  $$("[data-preset]").forEach(b => b.addEventListener("click", () => { setValues(PRESETS[b.dataset.preset].patch); baseline = null; run(); }));
  $("#load-critical").addEventListener("click", async () => {
    const r = await api("/api/orders?band=CRITICAL&limit=50");
    const pick = r.items[Math.floor(Math.random() * r.items.length)];
    go("simulate", new URLSearchParams({ from: pick.order_id }));
  });
  run();
}

// ------------------------------------------------------------------ BATCH
async function renderBatch() {
  view.innerHTML = `
    <p class="lede">Batch scoring uses the same live model configured in Settings as single-order scoring. Each row comes back <b>scored</b> (risk, band, reasons, suggested action) or <b>rejected</b> with the reason. Nothing is dropped or quietly repaired.</p>
    <div class="grid g2">
      <label class="dropzone" id="dz" for="file">
        <b>Drop a CSV here</b><span class="muted">or click to choose a file (up to 5,000 rows)</span>
        <input type="file" id="file" accept=".csv,text/csv" class="sr">
        <span class="btn primary">Choose file</span>
      </label>
      <section class="panel"><h2>What the file needs</h2>
        <p class="hint">One row per order. The configured dispatch model accepts the supplied order fields plus dispatch-time operational fields where available. Missing dispatch values are estimated and marked in the result; provide actual telemetry for operational use.</p>
        <div style="display:flex;gap:8px;flex-wrap:wrap"><a class="btn" href="/api/template.csv" download>Download template (10 example orders)</a>
          <button class="btn" id="try-template">Score the template now</button></div></section>
    </div>
    <div id="batch-out"></div>`;
  const dz = $("#dz");
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) upload(f); });
  $("#file").addEventListener("change", e => { const f = e.target.files[0]; if (f) upload(f); });
  $("#try-template").addEventListener("click", async () => {
    const text = await (await fetch("/api/template.csv")).text();
    upload(new File([text], "ontime_template.csv", { type: "text/csv" }));
  });
}

async function upload(file) {
  const out = $("#batch-out");
  out.innerHTML = `<div class="panel skeleton">Scoring ${esc(file.name)}…</div>`;
  const fd = new FormData(); fd.append("file", file);
  let r;
  try { r = await api("/api/score/batch", { method: "POST", body: fd }); }
  catch (e) { out.innerHTML = `<div class="callout bad"><div class="k">Could not score this file</div>${esc(e.message)}</div>`; return; }
  const s = r.summary;
  let filter = "all";
  const draw = () => {
    const rows = r.rows.filter(x => filter === "all" || (filter === "flagged" ? x.flagged : filter === "rejected" ? x.status === "REJECTED" : x.risk_level === filter));
    $("#b-body").innerHTML = rows.slice(0, 500).map(x => `<tr>
      <td class="mono">${esc(x.order_id ? shortId(String(x.order_id)) : "row " + (x.row + 1))}</td>
      <td>${x.status === "SCORED" ? pill(x.risk_level) : pill("REJECTED")}</td>
      <td class="num">${x.probability != null ? pct(x.probability, 1) : "—"}</td>
      <td>${x.status === "SCORED" ? esc(x.confidence === "LOW_CONFIDENCE" ? "Low confidence" : "Normal") : ""}</td>
      <td>${x.reasons ? esc(x.reasons) : `<span class="muted">—</span>`}</td>
      <td class="act">${x.status === "SCORED" ? action(x.suggested_action) : `<span class="err">${esc(x.reject_reasons)}</span>`}</td></tr>`).join("") || `<tr><td colspan="6" class="skeleton">No rows in this view.</td></tr>`;
  };
  out.innerHTML = `
    <div class="tiles">
      <div class="tile"><span class="k">Rows</span><span class="v">${fmt(s.total)}</span><span class="s">${esc(r.filename || "")}</span></div>
      <div class="tile"><span class="k">Scored</span><span class="v">${fmt(s.scored)}</span><span class="s">${fmt(s.low_confidence)} low confidence</span></div>
      <div class="tile"><span class="k">Rejected</span><span class="v" style="color:${s.rejected ? "var(--crit)" : "inherit"}">${fmt(s.rejected)}</span><span class="s">with the reason on each row</span></div>
      <div class="tile hl"><span class="k">Alerts</span><span class="v">${fmt(s.flagged)}</span><span class="s">${BANDS.map(b => `${b[0]}${b.slice(1).toLowerCase()} ${s.bands[b]}`).join(" · ")}</span></div>
    </div>
    <p class="note">Scoring model: <b>${esc(r.model || "configured model")}</b>. ${r.model === "synthetic_operational" ? "This is a prioritisation score, not a calibrated probability." : "The model identifier is returned by the backend for traceability."}</p>
    ${provNote(r.provenance)}
    ${s.rejected ? `<div class="callout bad" role="alert"><div class="k">${fmt(s.rejected)} row${s.rejected === 1 ? "" : "s"} could not be scored</div>
      <ul style="margin:4px 0 0 18px;padding:0">${Object.entries(r.error_reasons || {}).slice(0, 8).map(([k, v]) => `<li>${esc(k)} <span class="muted">(${fmt(v)} row${v === 1 ? "" : "s"})</span></li>`).join("")}</ul>
      <p class="note">First affected rows: ${(r.row_errors || []).slice(0, 8).map(x => `row ${x.row}${x.order_id ? " (" + esc(shortId(x.order_id)) + ")" : ""}`).join(", ")}${(r.row_errors || []).length > 8 ? "…" : ""}. Fix them in the file or remove them, then upload again.</p></div>` : ""}
    <section class="panel">
      <div class="panel-head"><div class="chips" id="b-filter">
        ${[["all", "All"], ["flagged", "Alerts"], ["CRITICAL", "Critical"], ["HIGH", "High"], ["rejected", "Rejected"]].map(([k, l], i) => `<button class="chip any" data-f="${k}" aria-pressed="${i === 0}">${l}</button>`).join("")}
      </div><button class="btn primary" id="b-dl">Download results (CSV)</button></div>
      <div class="table-wrap"><table><thead><tr><th>Order</th><th>Band</th><th class="num">Risk</th><th>Confidence</th><th>Reasons</th><th>Suggested action</th></tr></thead>
      <tbody id="b-body"></tbody></table></div>
      ${r.rows.length > 500 ? `<p class="note">Showing the first 500 rows; the download has all ${fmt(r.rows.length)}.</p>` : ""}
    </section>`;
  draw();
  $$("#b-filter [data-f]").forEach(b => b.addEventListener("click", () => { filter = b.dataset.f; $$("#b-filter [data-f]").forEach(x => x.setAttribute("aria-pressed", x === b)); draw(); }));
  $("#b-dl").addEventListener("click", () => {
    const cols = Object.keys(r.rows[0] || {});
    const pv = r.provenance || {};
    const extra = { model: pv.model, stage: pv.stage, data_origin: pv.data_origin, synthetic_benchmark: pv.synthetic };
    ["model", "stage", "data_origin", "synthetic_benchmark"].forEach(c => { if (!cols.includes(c)) cols.push(c); });
    r.rows.forEach(x => Object.keys(extra).forEach(k => { x[k] = extra[k]; }));
    const q = v => v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
    const csv = [cols.join(","), ...r.rows.map(x => cols.map(c => q(x[c])).join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = (r.filename || "orders").replace(/\.csv$/i, "") + "_scored.csv";
    document.body.appendChild(a); a.click(); a.remove();
  });
}

// ------------------------------------------------------------------ INSIGHTS
async function renderInsights() {
  const d = await api("/api/insights");
  const top = (arr, n) => arr.slice(0, n);
  view.innerHTML = `
    <p class="lede">Across <b>${fmt(d.n_orders)}</b> delivered orders, <b>${d.base_rate}%</b> arrived late. Lateness clusters by destination and season more than by product.</p>
    <div class="grid g2">
      <section class="panel"><h2>Late rate by customer state</h2><p class="hint">States with at least 100 orders, worst first.</p><div class="chart tall"><canvas id="i-state"></canvas></div></section>
      <section class="panel"><h2>Late rate by category</h2><p class="hint">Top 15 categories with at least 150 orders.</p><div class="chart tall"><canvas id="i-cat"></canvas></div></section>
      <section class="panel"><h2>Distance</h2><p class="hint">Straight-line seller-to-customer distance.</p><div class="chart sm"><canvas id="i-dist"></canvas></div>
        <p class="note">Beyond 1,500 km the late rate roughly doubles.</p></section>
      <section class="panel"><h2>Promised delivery window</h2><p class="hint">Short promises leave no slack.</p><div class="chart sm"><canvas id="i-prom"></canvas></div>
        <p class="note">Orders promised in under 10 days are late about 8× as often as those given 45+ days.</p></section>
      <section class="panel"><h2>Day the order was placed</h2><p class="hint">A small effect: early-week orders run slightly later.</p><div class="chart sm"><canvas id="i-dow"></canvas></div></section>
      <section class="panel"><h2>Same state vs. interstate</h2><p class="hint">Crossing a state line adds risk.</p><div class="chart sm"><canvas id="i-same"></canvas></div>
        <p class="note">Orders split across several sellers are rarely late (${d.split.map(x => `${x.key}: ${x.rate}%`).join(", ")}), so a split order is not a warning sign here.</p></section>
    </div>
    <section class="panel"><div class="panel-head"><div><h2>Seller watchlist</h2><p class="hint">The 25 sellers with the highest late rate among the ${fmt(d.n_sellers_30plus)} sellers with 30+ delivered orders.</p></div></div>
      <div class="table-wrap"><table><thead><tr><th>Seller</th><th>Location</th><th>Main category</th><th class="num">Orders</th><th class="num">Late</th><th></th></tr></thead>
      <tbody>${d.seller_watchlist.map(s => `<tr><td class="mono">${esc(shortId(s.seller_id))}</td><td>${esc(s.city)}, ${esc(s.state)}</td><td>${esc(nice(s.category))}</td>
        <td class="num">${fmt(s.n)}</td><td class="num"><b style="color:var(--crit)">${s.rate}%</b></td>
        <td><a class="btn sm" href="#/queue?q=${esc(s.seller_id)}">Orders in queue</a></td></tr>`).join("")}</tbody></table></div></section>`;
  const tip = (arr) => c => `${c.raw}% late · ${fmt(arr[c.dataIndex].n)} orders`;
  hbar($("#i-state"), d.state.map(s => s.key), d.state.map(s => s.rate), css("--chart-1"), { tip: tip(d.state), thick: 10 });
  hbar($("#i-cat"), top(d.category, 15).map(c => nice(c.key)), top(d.category, 15).map(c => c.rate), css("--chart-1"), { tip: tip(d.category) });
  vbar($("#i-dist"), d.distance.map(x => x.key), d.distance.map(x => x.rate), d.distance.map((x, i) => i === d.distance.length - 1 ? css("--accent") : css("--chart-2")), { tip: tip(d.distance) });
  vbar($("#i-prom"), d.promise.map(x => x.key), d.promise.map(x => x.rate), d.promise.map((x, i) => i === 0 ? css("--accent") : css("--chart-2")), { tip: tip(d.promise) });
  vbar($("#i-dow"), d.weekday.map(x => x.key), d.weekday.map(x => x.rate), css("--chart-2"), { tip: tip(d.weekday) });
  vbar($("#i-same"), d.same_state.map(x => x.key), d.same_state.map(x => x.rate), [css("--chart-2"), css("--accent")], { tip: tip(d.same_state) });
}

// ------------------------------------------------------------------ MODEL
const MODEL_NAMES = { logistic_regression: "Logistic regression", decision_tree: "Decision tree", random_forest: "Random forest",
  random_forest_tuned: "Random forest (tuned)", xgboost: "XGBoost", xgboost_tuned: "XGBoost (tuned)",
  xgboost_enriched: "XGBoost · enriched features", lightgbm_enriched: "LightGBM · enriched features", hist_gradient_enriched: "Histogram boosting · enriched features" };
const FI_LABEL = (f) => (S.meta.feature_labels[f]) || (f.startsWith("customer_state_") ? "Customer in " + f.slice(15)
  : f.startsWith("seller_state_") ? "Seller in " + f.slice(13) : f.startsWith("product_category_") ? "Category: " + nice(f.slice(17)) : nice(f));


// Reliability, stability and drift checks on the held-out real queue.
async function loadMonitoring() {
  const box = $("#mon-panel"); if (!box) return;
  let m;
  try { m = await api("/api/monitoring/summary"); }
  catch (e) { box.innerHTML = `<section class="panel"><h2>Reliability and monitoring</h2><p class="err">${esc(e.message)}</p></section>`; return; }
  if (!box.isConnected) return;   // the person left the page while this was loading
  const f3 = (v) => v == null ? "—" : Number(v).toFixed(3), pc = (v) => v == null ? "—" : (Number(v) * 100).toFixed(1) + "%";
  const ci = m.confidence_intervals, cm = ci ? ci.metrics : null;
  const row = (label, k, fmtr) => cm ? `<tr><td>${label}</td><td class="num"><b>${fmtr(cm[k].estimate)}</b></td><td class="num">${fmtr(cm[k].low)} to ${fmtr(cm[k].high)}</td></tr>` : "";
  const cal = m.calibration.calibrated_ensemble, raw = m.calibration.served_score;
  const calRows = (c) => c.bins.map(b => `<tr><td class="num">${b.bin}</td><td class="num">${fmt(b.n)}</td><td class="num">${pc(b.mean_score)}</td><td class="num">${pc(b.observed_late_rate)}</td></tr>`).join("");
  const mo = m.monthly_stability.months;
  const dr = m.drift_early_vs_late;
  box.innerHTML = `
    <section class="panel"><h2>How certain is the ${cm ? f3(cm.roc_auc.estimate) : "score"}? <span class="tag ${m.in_sample ? "" : "real"}">${m.in_sample ? "IN-SAMPLE" : "REAL ORDERS"}</span></h2>
      <p class="hint">${esc(m.provenance.data_origin)}. 95% bootstrap intervals from ${fmt(ci ? ci.bootstrap_samples : 0)} resamples of ${fmt(ci ? ci.n : 0)} orders (${fmt(ci ? ci.late : 0)} late), alerting on the riskiest ${pc(ci ? ci.alert_share : 0)}.</p>
      <div class="table-wrap"><table><thead><tr><th>Metric</th><th class="num">Estimate</th><th class="num">95% interval</th></tr></thead><tbody>
        ${row("ROC-AUC", "roc_auc", f3)}${row("PR-AUC", "pr_auc", f3)}${row("Precision of alerts", "precision", pc)}${row("Recall of alerts", "recall", pc)}</tbody></table></div></section>
    <section class="panel"><h2>Is the score a probability?</h2>
      <p class="hint">${esc(m.calibration.reading)}</p>
      <div class="tiles">
        <div class="tile"><span class="k">Served score: ECE</span><span class="v">${f3(raw.ece)}</span><span class="s">Brier ${f3(raw.brier)} vs ${f3(raw.brier_baseline)} for always guessing the base rate</span></div>
        ${cal ? `<div class="tile hl"><span class="k">Calibrated ensemble: ECE</span><span class="v">${f3(cal.ece)}</span><span class="s">Brier ${f3(cal.brier)} vs ${f3(cal.brier_baseline)} baseline: not better than guessing the base rate, so treat it as a ranking, not a probability</span></div>` : ""}
      </div>
      ${cal ? `<div class="table-wrap" style="margin-top:12px"><table><thead><tr><th class="num">Score bin</th><th class="num">Orders</th><th class="num">Mean calibrated score</th><th class="num">Observed late rate</th></tr></thead><tbody>${calRows(cal)}</tbody></table></div>` : ""}</section>
    <section class="panel"><h2>Does it hold up month by month?</h2>
      <p class="hint">${esc(m.monthly_stability.note)}</p>
      <div class="table-wrap"><table><thead><tr><th>Purchase month</th><th class="num">Orders</th><th class="num">Late rate</th><th class="num">ROC-AUC</th><th class="num">Precision</th><th class="num">Recall</th></tr></thead><tbody>
        ${mo.map(r => `<tr><td>${esc(r.month)}</td><td class="num">${fmt(r.orders)}</td><td class="num">${pc(r.late_rate)}</td><td class="num">${f3(r.roc_auc)}</td><td class="num">${pc(r.precision_at_budget)}</td><td class="num">${pc(r.recall_at_budget)}</td></tr>`).join("")}</tbody></table></div>
      <p class="note">ROC-AUC falls from the oldest to the newest month. Part of this can be an artefact (the newest orders only count if delivered by the data cut-off, which removes slow deliveries), but it is also a warning that the model must be re-validated on fresh outcomes before it is trusted in operation.</p></section>
    <section class="panel"><h2>Input drift inside the test period</h2>
      <p class="hint">Population Stability Index, first half of the held-out orders against the second half. ${esc(dr.thresholds)}.</p>
      <div class="table-wrap"><table><thead><tr><th>Input</th><th class="num">PSI</th><th>Status</th></tr></thead><tbody>
        ${dr.rows.map(r => `<tr><td>${esc(nice(r.feature))}</td><td class="num">${f3(r.psi)}</td><td>${esc(r.status || "—")}</td></tr>`).join("")}</tbody></table></div>
      <p class="note"><b>${esc(dr.recommendation)}</b></p>
      <h3 style="margin:16px 0 6px">Retraining criteria</h3><ul class="steps">${m.retraining_criteria.map(x => `<li><span>${esc(x)}</span></li>`).join("")}</ul>
      <p class="note"><b>Rollback:</b> ${esc(m.rollback)}</p>
      <div class="formrow" style="margin-top:12px">
        <div class="field"><label for="mon-drift-file">Check a new batch for drift (CSV of orders)</label><input id="mon-drift-file" type="file" accept=".csv"></div>
        <div class="field"><label for="mon-out-file">Upload real outcomes (CSV: order_id, is_delayed)</label><input id="mon-out-file" type="file" accept=".csv"></div></div>
      <div id="mon-result" aria-live="polite"></div></section>`;
  const post = async (url, file) => {
    const out = $("#mon-result"); out.innerHTML = `<span class="muted">Working…</span>`;
    const fd = new FormData(); fd.append("file", file);
    try { return await api(url, { method: "POST", body: fd }); }
    catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; return null; }
  };
  $("#mon-drift-file").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const d = await post("/api/monitoring/drift", f); if (!d) return;
    $("#mon-result").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Input</th><th class="num">PSI</th><th>Status</th></tr></thead><tbody>${d.rows.map(r => `<tr><td>${esc(nice(r.feature))}</td><td class="num">${f3(r.psi)}</td><td>${esc(r.status || "—")}</td></tr>`).join("")}</tbody></table></div><p class="note"><b>${esc(d.recommendation)}</b></p>`;
  });
  $("#mon-out-file").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const o = await post("/api/monitoring/outcomes", f); if (!o) return;
    if (o.status !== "ok") { $("#mon-result").innerHTML = `<div class="callout warn"><div class="k">Not enough to evaluate</div>${esc(o.message)} Matched ${fmt(o.matched_to_scored_orders)} of ${fmt(o.outcomes_rows)} rows.</div>`; return; }
    const c = o.ci.metrics;
    $("#mon-result").innerHTML = `<div class="tiles"><div class="tile hl"><span class="k">ROC-AUC on your outcomes</span><span class="v">${f3(c.roc_auc.estimate)}</span><span class="s">95%: ${f3(c.roc_auc.low)} to ${f3(c.roc_auc.high)}</span></div>
      <div class="tile"><span class="k">Alert precision</span><span class="v">${pc(o.alert_precision)}</span><span class="s">${fmt(o.alerts)} alerts</span></div>
      <div class="tile"><span class="k">Alert recall</span><span class="v">${pc(o.alert_recall)}</span><span class="s">${fmt(o.late)} late of ${fmt(o.matched_to_scored_orders)} matched</span></div></div>`;
  });
}

async function renderModel() {
  const [m, ev, th, cmp] = await Promise.all([
    api("/api/model"),
    api("/api/model/evaluation").catch(() => null),
    api("/api/model/thresholds").catch(() => null),
    api("/api/models/compare").catch(() => null),
  ]);
  const n3 = (v) => (v == null || !Number.isFinite(Number(v)) ? "—" : Number(v).toFixed(3));
  const p1 = (v) => (v == null || !Number.isFinite(Number(v)) ? "—" : (Number(v) * 100).toFixed(1) + "%");
  const comparePanel = !cmp ? "" : `
    <section class="panel"><h2>Which algorithms does OnTime use?</h2>
      <p class="hint">Five algorithm families were built or tried. A <b>linear</b> model draws one straight-line weight per input; <b>tree</b> models split the data into branches and capture interactions; an <b>ensemble</b> blends several models so their mistakes partly cancel.</p>
      <div class="table-wrap"><table class="cmp"><thead><tr><th>Algorithm</th><th>Type</th><th>How it works</th><th class="num">Weight in blend</th></tr></thead><tbody>
      ${cmp.families.map(f => `<tr><td><b>${esc(f.name)}</b></td><td><span class="tag">${esc(f.kind)}</span></td><td>${esc(f.how)} <span class="muted">${esc(f.strength)}</span></td>
        <td class="num">${f.in_ensemble ? `<div class="wbar" title="${Math.round(f.ensemble_weight * 100)}%"><i style="width:${Math.round(f.ensemble_weight * 100)}%"></i></div><b>${Math.round(f.ensemble_weight * 100)}%</b>` : '<span class="muted">not in blend</span>'}</td></tr>`).join("")}
      </tbody></table></div>
      ${cmp.ensemble.decision ? `<p class="note"><b>Why the blend?</b> ${esc(cmp.ensemble.decision)}</p>` : ""}</section>
    <section class="panel"><h2>Model comparison</h2>
      <p class="hint">Same question, different models and different tests. Compare rows only when the <b>test</b> column matches: a random split mixes past and future orders, so it reads higher than a chronological one.</p>
      <div class="table-wrap"><table class="cmp"><thead><tr><th>Model</th><th>Stage</th><th>Data</th><th>Test</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">Precision</th><th class="num">Recall</th></tr></thead><tbody>
      ${cmp.scoreboard.map(r => `<tr class="${r.serving_queue ? "serving" : ""}"><td><b>${esc(r.label)}</b>${r.serving_queue ? ' <span class="tag">SERVING QUEUE</span>' : ""}</td><td>${esc(r.stage)}</td>
        <td><span class="tag ${r.real_data ? "real" : ""}">${r.real_data ? "Real orders" : "Generated data"}</span></td><td class="small">${esc((r.protocol || "").split(".")[0])}</td>
        <td class="num"><b>${n3(r.roc_auc)}</b></td><td class="num">${n3(r.pr_auc)}</td><td class="num">${p1(r.precision)}</td><td class="num">${p1(r.recall)}</td></tr>`).join("")}
      </tbody></table></div>
      ${cmp.candidates.length ? `<h3 style="margin:18px 0 6px">Candidates tried for the risk queue (validation ROC-AUC)</h3>
        ${cmp.candidates.map(c => `<div class="crow"><span>${esc(c.name)}${c.selected ? ' <span class="tag">chosen</span>' : ""}</span><div class="wbar"><i style="width:${Math.round(c.validation_auc * 100)}%"></i></div><b>${n3(c.validation_auc)}</b></div>`).join("")}
        <p class="note">XGBoost won on validation data, so it serves the queue.</p>` : ""}</section>`;
  const first = ev && (ev.active || (ev.rows && ev.rows[0]));
  const inSample = !!(first && first.in_sample);
  const realPanel = !first ? "" : `
    ${viewNotice()}
    ${ev.dispatch_selected && ev.dispatch ? `<section class="panel"><h2>Dispatch model benchmark <span class="tag">GENERATED DATA</span></h2>
      <p class="hint">${esc(ev.dispatch.protocol)}</p>
      <div class="tiles"><div class="tile hl"><span class="k">ROC-AUC (generated)</span><span class="v">${n3(ev.dispatch.roc_auc)}</span><span class="s">Not real-world performance</span></div>
        <div class="tile"><span class="k">PR-AUC (generated)</span><span class="v">${n3(ev.dispatch.pr_auc)}</span><span class="s">Generated positive class</span></div>
        <div class="tile"><span class="k">Precision at 0.5</span><span class="v">${p1(ev.dispatch.precision)}</span><span class="s">Generated data</span></div>
        <div class="tile"><span class="k">Recall at 0.5</span><span class="v">${p1(ev.dispatch.recall)}</span><span class="s">Generated data</span></div></div>
      <p class="note">${esc(ev.dispatch.note)} The queue below stays on the real-Olist model.</p></section>` : ""}
    <section class="panel"><h2>How well does the risk queue work on real orders? <span class="tag ${inSample ? "" : "real"}">${esc(first.label)}</span></h2>
      <p class="hint">${esc(first.protocol)} ${inSample ? "Held-out figures from the model's own random split are shown." : "This is the number to quote for real-world performance."}</p>
      <div class="tiles">
        <div class="tile hl"><span class="k">ROC-AUC (${inSample ? "own random split" : "real orders"})</span><span class="v">${n3(first.roc_auc)}</span><span class="s">0.5 is a coin flip, 1.0 is perfect</span></div>
        <div class="tile"><span class="k">PR-AUC</span><span class="v">${n3(first.pr_auc)}</span><span class="s">Late orders are rare (${p1(ev.queue.late_rate)} of the newest orders)${inSample ? "" : ", so this is low"}</span></div>
        <div class="tile"><span class="k">Precision of alerts</span><span class="v">${p1(first.precision)}</span><span class="s">Share of alerts that were really late</span></div>
        <div class="tile"><span class="k">Recall of alerts</span><span class="v">${p1(first.recall)}</span><span class="s">Share of late orders the alerts caught</span></div>
      </div>
      ${inSample ? `<div class="tiles" style="margin-top:12px"><div class="tile"><span class="k">ROC-AUC on the newest orders (in-sample)</span><span class="v">${n3(first.queue_roc_auc)}</span><span class="s">Optimistic: most of these orders were in training</span></div>
        <div class="tile"><span class="k">PR-AUC (in-sample)</span><span class="v">${n3(first.queue_pr_auc)}</span><span class="s">Optimistic</span></div>
        <div class="tile"><span class="k">Alert precision (in-sample)</span><span class="v">${p1(first.queue_precision)}</span><span class="s">${fmt(first.queue_alerts || 0)} alerts</span></div>
        <div class="tile"><span class="k">Alert recall (in-sample)</span><span class="v">${p1(first.queue_recall)}</span><span class="s">Optimistic</span></div></div>` : ""}
      <p class="note">${esc(first.note)}</p>
    </section>
    ${th && th.rows ? `<section class="panel"><h2>Alert size: how many orders to check</h2>
      <p class="hint">Measured on the same ${fmt(th.n)} ${th.in_sample ? "newest" : "unseen"} orders as the Overview, ranked by ${esc(th.model_label || "the active model")}. Checking more orders catches more late deliveries but costs more work.</p>
      <div class="table-wrap"><table><thead><tr><th class="num">Check the riskiest</th><th class="num">Orders</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th></tr></thead>
      <tbody>${th.rows.map(r => `<tr><td class="num">${Math.round(r.alert_share * 100)}%</td><td class="num">${fmt(r.alerts)}</td><td class="num">${p1(r.precision)}</td><td class="num">${p1(r.recall)}</td><td class="num">${n3(r.f1)}</td></tr>`).join("")}</tbody></table></div>
      <p class="note">${esc(th.note)}</p></section>` : ""}
    <section class="panel"><h2>Every score and what its AUC really means</h2>
      <p class="hint">${esc(ev.reading_guide)} Scores from a random split read higher than scores from a chronological split; compare like with like.</p>
      <div class="table-wrap"><table><thead><tr><th>Score</th><th>Used when</th><th>Data</th><th class="num">ROC-AUC</th><th class="num">Precision</th><th class="num">Recall</th><th>How it was tested</th></tr></thead>
      <tbody>${ev.rows.map(r => `<tr><td><b>${esc(r.label)}</b>${r.serving_queue ? ' <span class="tag">SERVING QUEUE</span>' : ""}</td><td>${esc(r.stage)}</td><td>${r.real_data ? esc(r.data) : ` ${esc(r.data)}`}</td><td class="num">${n3(r.roc_auc)}</td><td class="num">${p1(r.precision)}</td><td class="num">${p1(r.recall)}</td><td>${esc(r.protocol)}<br><span class="muted">${esc(r.note)}</span></td></tr>`).join("")}</tbody></table></div>
    </section>`;
  const live = m.live_scoring || {};
  const modelKey = live.key || S.meta.live_scoring_model || "synthetic_operational";
  const modelLabel = live.label || S.meta.live_scoring_label || "Dispatch operational model";
  const isSynthetic = modelKey === "synthetic_operational";
  const available = live.available !== false;
  const features = Array.isArray(live.benchmark_top_features) ? live.benchmark_top_features : [];
  view.innerHTML = `
    <p class="lede">Live single-order and batch scoring are configured to use <b>${esc(modelLabel)}</b>. This page reflects the model configured in Settings and the same model identifier returned by the scoring API.</p>
    <div class="tiles">
      <div class="tile hl"><span class="k">Live scoring model</span><span class="v" style="font-size:20px">${esc(modelLabel)}</span><span class="s">${available ? "Loaded and available" : "Not available — check model files and dependencies"}</span></div>
      <div class="tile"><span class="k">Scoring threshold</span><span class="v">${live.threshold == null ? "—" : Number(live.threshold).toFixed(2)}</span><span class="s">Read from backend model configuration</span></div>
      <div class="tile"><span class="k">Scoring stage</span><span class="v" style="font-size:20px">${isSynthetic ? "Dispatch time" : "Configured model"}</span><span class="s">Single-order and batch requests use the same setting</span></div>
    </div>
    ${isSynthetic ? `
      <section class="panel"><h2>Dispatch model benchmark</h2>
        <div class="callout warn"><div class="k">Generated benchmark, not real-world performance</div>Olist does not publish dispatch, tracking, weather, congestion or carrier-capacity feeds. These signals and the late/on-time labels in this dataset were <b>generated</b> from Olist-derived distributions. The result shows the pipeline works on that data; it must be re-measured on real dispatch outcomes. The labels were drawn from a formula over these same dispatch signals, so the model is largely recovering the simulator's rules, and the training script was set to require an AUC above 0.91. Do not compare it with the checkout-time score (ROC-AUC 0.717 on real orders).</div>
        <p class="hint">The dispatch model was evaluated on an untouched chronological test set of the generated operational dataset.</p>
        <div class="tiles">
          <div class="tile hl"><span class="k">Held-out ROC-AUC</span><span class="v">${live.benchmark_roc_auc == null ? "—" : Number(live.benchmark_roc_auc).toFixed(3)}</span><span class="s">Chronological synthetic test set</span></div>
          <div class="tile"><span class="k">Held-out PR-AUC</span><span class="v">${live.benchmark_pr_auc == null ? "—" : Number(live.benchmark_pr_auc).toFixed(3)}</span><span class="s">Synthetic positive class</span></div>
          <div class="tile"><span class="k">Training / validation / test</span><span class="v">${fmt(live.benchmark_train_rows || 0)} / ${fmt(live.benchmark_validation_rows || 0)} / ${fmt(live.benchmark_test_rows || 0)}</span><span class="s">Separate chronological partitions</span></div>
        </div>
        <div class="table-wrap" style="margin-top:12px"><table><thead><tr><th>Dispatch feature</th><th class="num">Model gain</th><th class="num">Held-out AUC drop</th></tr></thead><tbody>${features.slice(0,8).map(x => `<tr><td>${esc(x.feature)}</td><td class="num">${Number(x.gain_importance || 0).toFixed(3)}</td><td class="num">${Number(x.test_permutation_auc_drop || 0).toFixed(4)}</td></tr>`).join("")}</tbody></table></div>
        <p class="note">Missing dispatch inputs are marked as estimates during scoring. See the data note on the How it works page.</p>
      </section>` : `
      <section class="panel"><h2>Model configuration</h2>
        <p class="hint">The live scoring model has been selected in Settings. The dispatch benchmark appears on this page only when the dispatch operational model is selected in Settings. The backend's scoring response includes the actual model identifier and label for each request.</p>
      </section>`}
    ${realPanel}
    <div id="mon-panel"><section class="panel skeleton">Loading reliability and monitoring checks…</section></div>
    ${comparePanel}
    <section class="panel"><h2>Live weather lookup <span class="tag">LIVE API</span></h2>
      <p class="hint">Fetch current conditions and a three-day forecast for a delivery-point coordinate. This uses Open-Meteo live data and provides operational context; weather is not retroactively inserted into historical order records.</p>
      <div class="formrow"><div class="field"><label for="wx-lat">Latitude</label><input id="wx-lat" type="number" min="-90" max="90" step="any" placeholder="e.g. -23.55" value="-23.55"></div>
      <div class="field"><label for="wx-lon">Longitude</label><input id="wx-lon" type="number" min="-180" max="180" step="any" placeholder="e.g. -46.63" value="-46.63"></div>
      <div class="field" style="align-self:end"><button class="btn" id="wx-load" type="button">Fetch live forecast</button></div></div>
      <p class="note">Example coordinates are for São Paulo, Brazil. Change them to the delivery location you want to check.</p>
      <div id="m-live-weather" class="table-wrap"><span class="muted">Weather is not fetched until you click the button.</span></div></section>
    <section class="panel"><h2>How scoring stays in sync</h2>
      <ul class="steps">
        <li><b>One setting</b><span>Single-order and batch scoring use the same configured model unless a request explicitly overrides it.</span></li>
        <li><b>Model identity</b><span>Every score response includes its model key, label, threshold, warnings and estimated inputs.</span></li>
        <li><b>Separate historical view</b><span>The existing order queue is based on stored historical records. The app does not invent missing dispatch telemetry to rescore those records.</span></li>
        <li><b>Operational validation</b><span>Track predictions against actual dispatch and delivery outcomes before treating scores as calibrated probabilities.</span></li>
      </ul>
    </section>`;
  loadMonitoring();
  const wxButton = $("#wx-load");
  if (wxButton) wxButton.addEventListener("click", async () => {
    const box = $("#m-live-weather");
    const lat = Number($("#wx-lat").value), lon = Number($("#wx-lon").value);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      box.innerHTML = `<p class="error">Enter a valid latitude (-90 to 90) and longitude (-180 to 180).</p>`; return;
    }
    box.innerHTML = `<span class="muted">Fetching live forecast…</span>`;
    try {
      const w = await api(`/api/weather/forecast?latitude=${encodeURIComponent(lat)}&longitude=${encodeURIComponent(lon)}`);
      const c = w.current || {}, d = w.daily || {};
      const desc = code => ({0:"Clear",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",48:"Depositing rime fog",51:"Light drizzle",53:"Drizzle",55:"Heavy drizzle",61:"Light rain",63:"Rain",65:"Heavy rain",71:"Light snow",73:"Snow",75:"Heavy snow",80:"Rain showers",81:"Rain showers",82:"Heavy rain showers",95:"Thunderstorm",96:"Thunderstorm with hail",99:"Thunderstorm with hail"})[Number(code)] || `Weather code ${code}`;
      const days = (d.time || []).map((date, i) => `<tr><td>${esc(date)}</td><td>${esc(desc((d.weather_code || [])[i]))}</td><td class="num">${(d.temperature_2m_min || [])[i] ?? "—"}° / ${(d.temperature_2m_max || [])[i] ?? "—"}°C</td><td class="num">${(d.precipitation_probability_max || [])[i] ?? "—"}%</td><td class="num">${(d.wind_speed_10m_max || [])[i] ?? "—"} km/h</td></tr>`).join("");
      box.innerHTML = `<p><b>${esc(w.source)}</b> · ${esc(w.timezone || "local timezone")} · ${esc(c.time || "current conditions")}</p>
        <div class="tiles"><div class="tile"><span class="k">Now</span><span class="v">${c.temperature_2m ?? "—"}°C</span><span class="s">Feels like ${c.apparent_temperature ?? "—"}°C · ${esc(desc(c.weather_code))}</span></div>
        <div class="tile"><span class="k">Humidity</span><span class="v">${c.relative_humidity_2m ?? "—"}%</span><span class="s">Current relative humidity</span></div>
        <div class="tile"><span class="k">Precipitation</span><span class="v">${c.precipitation ?? "—"} mm</span><span class="s">Current</span></div>
        <div class="tile"><span class="k">Wind</span><span class="v">${c.wind_speed_10m ?? "—"} km/h</span><span class="s">Current</span></div></div>
        <div class="table-wrap"><table><thead><tr><th>Date</th><th>Forecast</th><th class="num">Min / Max</th><th class="num">Precip. chance</th><th class="num">Max wind</th></tr></thead><tbody>${days}</tbody></table></div>
        <p class="note">${esc(w.note || "Live forecast; not used in the trained model.")}</p>`;
    } catch (e) { box.innerHTML = `<p class="error">${esc(e.message || "Live weather is unavailable. No synthetic weather was substituted.")}</p>`; }
  });
}

// ------------------------------------------------------------------ ABOUT
async function renderAbout() {
  view.innerHTML = `
    <section class="panel"><h2>The problem</h2><div class="prose">
      <p>Every Olist order gets a promised delivery date at checkout. About 8% miss it, by more than 9 days on average. Most teams only find out after the date has passed, when all that's left is an apology.</p>
      <p>OnTime combines a historical order-risk queue with a live dispatch-time scoring workflow. The live single-order and batch scorer uses the model configured in Settings and returns its model identity, risk ranking, explanations and any estimated inputs.</p></div></section>
    <section class="panel"><h2>From raw data to an action</h2>
      <ol class="steps">
        <li><b>Clean and join</b><span>Nine Olist tables become one row per delivered order: 96,470 orders, 44 columns.</span></li>
        <li><b>Explore</b><span>Late rates by month, state, distance, promise window and seller.</span></li>
        <li><b>Build features</b><span>33 order-time features, including leak-free past late rates for the seller, category and region.</span></li>
        <li><b>Train and validate</b><span>Separate checkout-time and dispatch-time workflows use their matching feature sets and evaluation protocols.</span></li>
        <li><b>Score and explain</b><span>Risk ranking, risk band and the main factors driving each live score; estimated inputs are identified.</span></li>
        <li><b>Act</b><span>A suggested action per order, and a shared board to track what the team did.</span></li>
      </ol></section>
    <div class="grid g2">
      <section class="panel"><h2>Reading a score</h2>
        <dl class="kv" style="font-size:15px"><dt>${pill("LOW")}</dt><dd>under 30% · no action</dd><dt>${pill("MEDIUM")}</dt><dd>30–60% · monitor</dd>
          <dt>${pill("HIGH")}</dt><dd>60–80% · alert, act today</dd><dt>${pill("CRITICAL")}</dt><dd>80%+ · urgent</dd></dl>
        <p class="note">For live dispatch scoring, the alert threshold and selected model are read from the backend configuration. The score is a prioritisation signal; it should not be interpreted as a calibrated real-world probability without validation.</p></section>
      <section class="panel"><h2>Limits to keep in mind</h2><div class="prose">
        <p><b>Validate alerts before automating actions.</b> A person should review high-impact decisions until the model is checked against current operational outcomes.</p>
        <p><b>Operational inputs matter.</b> Weather, route disruption, carrier reliability and hub backlog should come from actual dispatch systems where available.</p>
        <p><b>Data note.</b> Real carrier, weather and tracking feeds were not available, so the operational dataset behind the dispatch model is generated. The Olist order history is real. Results on the generated data show the pipeline works; live accuracy needs real outcomes.</p>
        <p><b>Monitor drift.</b> Re-evaluate the model as delivery patterns, carriers and service levels change.</p></div></section>
    </div>
    <section class="panel"><h2>What the review asked for, and where it is</h2>
      <p class="hint">From the project review on 2 October 2026.</p>
      <div class="table-wrap"><table><thead><tr><th>Mentor's suggestion</th><th>What OnTime does</th><th>Where</th></tr></thead><tbody>
        <tr><td>Use e-commerce data, but focus on 5-6 product types; a late healthcare product is not the same as a late cushion</td><td>Six focus categories with a criticality tier, impact statement, playbook and a priority sort (risk × category weight)</td><td><a href="#/focus">Focus categories</a></td></tr>
        <tr><td>Form clusters of delayed orders with the same pattern (vendor, area, product) and understand why</td><td>K-means groups 7,826 late orders into 6 patterns, profiled by what sets them apart, with lift-based states and products</td><td><a href="#/patterns">Delay patterns</a></td></tr>
        <tr><td>Don't just surface problems: suggest how they could be solved now and in future</td><td>Every pattern gets specific recommendations (e.g. minimum promise days per route); every order gets a suggested action</td><td><a href="#/patterns">Patterns</a>, <a href="#/queue">Risk queue</a></td></tr>
        <tr><td>Narrative reporting: tell the story of a cluster or a single record</td><td>Written stories for each order, pattern, category and period, built from real numbers; optional AI rewrite</td><td>Order drawer, <a href="#/briefing">Briefing</a></td></tr>
        <tr><td>Look at delivery models (hub-and-spoke etc.); fill gaps in the data</td><td>Delivery model, hub transfers, shipping mode, carrier and hubs, with a sensitivity test</td><td><a href="#/network">Network</a></td></tr>
        <tr><td>A periodic dashboard check is more practical than real-time alerts</td><td>A monthly briefing with patterns, focus categories, critical orders and recommended focus; downloadable</td><td><a href="#/briefing">Briefing</a></td></tr>
      </tbody></table></div></section>
    <section class="panel"><h2>Roadmap</h2><div class="prose">
      <p><b>Next:</b> real carrier and tracking scans instead of simulated logistics; road distance from a maps API; holiday and weather calendars; monthly automatic retraining with drift alerts.</p>
      <p><b>Later:</b> quick-commerce data (different buying patterns and much shorter promises), the remaining product categories' playbooks, and an API the order system calls at checkout.</p></div></section>
    <section class="panel"><h2>For developers</h2><div class="prose">
      <p>The backend is FastAPI; interactive API docs are at <a href="/docs">/docs</a>. Every route is used by this page: <span class="mono">/api/summary</span>, <span class="mono">/api/orders</span>, <span class="mono">/api/score</span>, <span class="mono">/api/score/batch</span>, <span class="mono">/api/board</span>, <span class="mono">/api/insights</span>, <span class="mono">/api/model</span>, <span class="mono">/api/weather/forecast</span> (live Open-Meteo), <span class="mono">/api/handoff/model</span>, <span class="mono">/api/tracking/model</span> (simulated).</p></div></section>
    <section class="panel"><h2>Prediction stages</h2><p>The historical queue and live dispatch scorer operate at different information stages. Dispatch-time inputs are required for the operational model; missing operational values are estimated and explicitly flagged. Simulated weather, tracking and disruption fields are labelled as simulated and must not be treated as observed delivery history.</p></section>`;
}

// ------------------------------------------------------------------ PATTERNS
const H4S = 'style="margin:0 0 10px;font:600 17px var(--display);letter-spacing:.02em"';
const kicker = (t) => `<div style="font:600 11.5px var(--body);text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:6px">${t}</div>`;
function liftChips(rows, fmtKey = (k) => k) {
  return rows && rows.length ? `<div class="chips2">${rows.map(r => `<span class="dchip" title="${r.share}% of this pattern">${esc(fmtKey(r.key))} · ${r.share}% · ${r.lift}×</span>`).join("")}</div>` : `<span class="muted">No strong concentration</span>`;
}
function zBars(z) {
  const names = { distance_km: "Route distance", promised_days: "Promised window", seller_late_rate: "Seller's past late rate", peak_season: "Peak-season share",
    weight_kg: "Parcel weight", order_value: "Order value", freight_share: "Freight share", same_state: "Same-state share", approval_hours: "Approval time" };
  const max = Math.max(2, ...Object.values(z).map(Math.abs));
  return `<div class="zbars">${Object.entries(z).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, v]) => {
    const w = Math.abs(v) / max * 50;
    return `<div class="row"><span>${names[k] || k}</span><span class="track"><i style="${v >= 0 ? `left:50%;width:${w}%;background:var(--crit)` : `right:50%;width:${w}%;background:var(--low)`}"></i></span><span class="v">${v > 0 ? "+" : ""}${v.toFixed(2)}</span></div>`;
  }).join("")}</div><p class="note">Bars show how far this pattern sits from late orders in general (standard deviations). Red = higher than usual, green = lower.</p>`;
}

async function renderPatterns(params) {
  const data = await api("/api/patterns");
  S.patternNames = data.patterns.map(p => ({ rank: p.rank, name: p.name }));
  const rank = +params.get("p");
  if (rank) return renderPatternDetail(rank, data);
  const sil = data.silhouette_by_k[data.k];
  view.innerHTML = `
    <p class="lede">Instead of chasing ${fmt(data.n_late)} late orders one by one, OnTime groups them into <b>${data.patterns.length} patterns</b> that share a cause:
      the same kind of route, season, seller or parcel. Each pattern comes with its own story and its own fix.</p>
    <div class="pcards">${data.patterns.map(p => `
      <button class="pcard" data-p="${p.rank}" type="button">
        <div class="top"><span class="rank">P${p.rank}</span><span class="muted" style="font-size:13px">${fmt(p.n)} late orders · ${p.share_of_late}%</span></div>
        <h3>${esc(p.name)}</h3>
        <div class="share"><i style="width:${Math.min(100, p.share_of_late * 3)}%"></i></div>
        <div class="chips2">${p.distinguishing.slice(0, 3).map(d => `<span class="dchip ${d.z > 0 ? "up" : "down"}">${esc(d.label)}</span>`).join("")}</div>
        <div class="meta">Promised ${p.median_promised_days.toFixed(0)} days, took ${p.median_actual_days.toFixed(0)} · ${fmt(p.median_distance_km)} km median · ${fmt(p.queue_alerts)} alerts in the queue now</div>
        <div class="meta" style="color:var(--ink)"><b>First fix:</b> ${esc(p.recommendations[0])}</div>
      </button>`).join("")}</div>
    <section class="panel"><h2>How the patterns were found</h2><div class="prose">
      <p>Only orders that really arrived late were grouped. Each is described by things a logistics team can act on: route distance, promised window,
        the seller's past late rate, peak-season timing, weight, value, freight share, same-state delivery, payment approval time and destination region.
        K-means tried 4 to 7 groups and kept ${data.k}, which separated them best (silhouette ${sil}).</p>
      <p>That score is modest: late orders don't split into perfectly separate types, so treat the patterns as useful working groups, not hard categories.
        Patterns are described by what makes them <i>different</i> from late orders in general. States and products are listed by lift, so a big state like SP or RJ
        only shows up where it is really over-represented.</p></div></section>`;
  $$("[data-p]").forEach(b => b.addEventListener("click", () => go("patterns", new URLSearchParams({ p: b.dataset.p }))));
}

async function renderPatternDetail(rank, all) {
  const p = await api(`/api/patterns/${rank}`);
  const lg = p.logistics;
  view.innerHTML = `
    <div class="no-print"><a class="btn ghost" href="#/patterns">← All patterns</a></div>
    <div class="hero">
      <section class="hero-main">
        ${kicker(`Pattern ${p.rank} of ${all.patterns.length} · ${fmt(p.n)} late orders · ${p.share_of_late}% of all late deliveries`)}
        <h2>${esc(p.name)}</h2>
        <div class="story" id="p-story">${p.story.map(x => `<p>${esc(x)}</p>`).join("")}</div>
        <div class="story-tools no-print"><button class="btn sm" id="p-copy">Copy story</button><button class="btn sm" id="p-rw">Rewrite for a manager</button><span id="p-eng"></span></div>
      </section>
      <section class="panel"><h2>What to do</h2><p class="hint">Specific to this pattern, from what sets it apart.</p>
        <ol class="recs">${p.recommendations.map(r => `<li><span>${esc(r)}</span></li>`).join("")}</ol></section>
    </div>
    <div class="mini-kpis">
      <div><b>${fmt(p.median_distance_km)} km</b><span>median route</span></div>
      <div><b>${p.median_promised_days.toFixed(0)} → ${p.median_actual_days.toFixed(0)} d</b><span>promised → actual</span></div>
      <div><b>${p.median_days_late.toFixed(0)} d</b><span>median days late</span></div>
      <div><b>${p.peak_share.toFixed(0)}%</b><span>placed in peak months</span></div>
      <div><b>${p.median_weight_kg.toFixed(1)} kg</b><span>median weight</span></div>
      <div><b>R$ ${fmt(p.median_order_value)}</b><span>median order value</span></div>
      <div><b>${fmt(p.queue_alerts)}</b><span>alerts in the queue now</span></div>
    </div>
    <div class="grid g2">
      <section class="panel"><h2>What sets it apart</h2>${zBars(p.z)}</section>
      <section class="panel"><h2>Where and what</h2>
        ${kicker("Destination regions (share · lift)")}${liftChips(p.regions)}
        <div style="height:12px"></div>${kicker("States")}${liftChips(p.states)}
        <div style="height:12px"></div>${kicker("Products")}${liftChips(p.categories, nice)}
        <div style="height:12px"></div>${kicker("Sellers")}<span>${fmt(p.top_sellers.n_sellers)} sellers; the top 10 account for ${p.top_sellers.top10_share}%.</span>
        <div style="height:14px"></div>${kicker("Logistics view")}
        <div class="chips2">${Object.entries(lg.delivery_model).map(([k, v]) => `<span class="dchip">${esc(k)} ${v}%</span>`).join("")}</div>
        <p class="note">Median ${lg.median_transfers} hub transfer(s); first mile ${esc(lg.main_first_mile_hub)}, last mile ${esc(lg.main_last_mile_hub)}.</p></section>
    </div>
    <section class="panel"><div class="panel-head"><div><h2>Alerts in the queue that match this pattern</h2><p class="hint">Highest risk first.</p></div>
      <a class="btn" href="#/queue?pattern=${p.rank}&flagged=true">Open in risk queue</a></div>
      <div class="table-wrap"><table><thead><tr><th>Order</th><th>Band</th><th class="num">Risk</th><th>Category</th><th>Route</th><th>Suggested action</th></tr></thead>
      <tbody>${p.orders.map(o => `<tr class="clickable" data-id="${esc(o.order_id)}"><td class="mono">${esc(shortId(o.order_id))}</td><td>${pill(o.risk_level)}</td>
        <td class="num">${pct(o.probability, 1)}</td><td>${esc(nice(o.product_category))}</td><td>${esc(o.seller_state)} → ${esc(o.customer_state)}</td><td class="act">${action(o.suggested_action)}</td></tr>`).join("")
        || `<tr><td colspan="6" class="skeleton">No alerts match this pattern right now.</td></tr>`}</tbody></table></div></section>`;
  $$("tr[data-id]").forEach(tr => tr.addEventListener("click", () => openOrder(tr.dataset.id)));
  $("#p-copy").addEventListener("click", () => copyText($("#p-story").innerText));
  $("#p-rw").addEventListener("click", () => rewriteInto($("#p-story"), "executive", $("#p-eng"), $("#p-rw")));
}

// ------------------------------------------------------------------ FOCUS CATEGORIES
async function renderFocus(params) {
  const d = await api("/api/categories");
  const key = params.get("k") || d.focus[0].key;
  const c = await api(`/api/categories/${key}`);
  const st = c.stats;
  const checks = store.get("ontime-playbook-" + key, {});
  view.innerHTML = `
    <p class="lede">The mentor's advice was to be sharp rather than broad: six product types where a late delivery means something different, from essentials
      (health, baby) to deadline-sensitive gifts and bulky furniture. Every other category is still scored; this is where the deeper analysis is.</p>
    <div class="ctabs">${d.focus.map(f => `<button class="ctab" type="button" data-k="${f.key}" aria-pressed="${f.key === key}">
      <span class="tier ${esc(f.tier.replace(/ /g, "-"))}">${esc(f.tier)}</span><b>${esc(f.label)}</b>
      <span class="rate" style="color:${f.stats.late_rate > f.stats.platform_rate ? "var(--crit)" : "var(--low)"}">${f.stats.late_rate.toFixed(1)}%</span>
      <small>late · ${fmt(f.queue.alerts)} alerts in queue</small></button>`).join("")}</div>
    <div class="grid g-main">
      <section class="panel"><h2>${esc(c.label)}</h2>
        <div class="story" id="f-story">${c.story.map(x => `<p>${esc(x)}</p>`).join("")}</div>
        <div class="story-tools no-print"><button class="btn sm" id="f-copy">Copy</button><button class="btn sm" id="f-rw">Rewrite for a manager</button><span id="f-eng"></span></div></section>
      <section class="panel"><h2>Playbook</h2><p class="hint">Tick what the team has put in place. Saved in this browser.</p>
        <ul class="checklist">${c.playbook.map((x, i) => `<li><input type="checkbox" id="pb-${i}" ${checks[i] ? "checked" : ""}><label for="pb-${i}">${esc(x)}</label></li>`).join("")}</ul>
        <div class="mini-kpis" style="margin-top:16px">
          <div><b>${fmt(st.n)}</b><span>delivered orders</span></div><div><b>${st.avg_days_late} d</b><span>avg lateness when late</span></div>
          <div><b>R$ ${fmt(st.median_value)}</b><span>median order value</span></div><div><b>${st.median_weight_kg} kg</b><span>median weight</span></div></div></section>
    </div>
    <div class="grid g2">
      <section class="panel"><h2>Late rate by month</h2><p class="hint">This category against the platform average (${st.platform_rate}%).</p><div class="chart"><canvas id="f-month"></canvas></div></section>
      <section class="panel"><h2>Which delay patterns hit it</h2><p class="hint">Share of this category's late orders in each pattern.</p><div class="chart"><canvas id="f-pat"></canvas></div></section>
      <section class="panel"><h2>Worst destinations</h2><p class="hint">States with at least 40 orders in this category.</p><div class="chart sm"><canvas id="f-st"></canvas></div></section>
      <section class="panel"><h2>Alerts in the queue</h2><div class="table-wrap"><table><thead><tr><th>Order</th><th>Band</th><th class="num">Risk</th><th>Route</th></tr></thead>
        <tbody>${c.orders.map(o => `<tr class="clickable" data-id="${esc(o.order_id)}"><td class="mono">${esc(shortId(o.order_id))}</td><td>${pill(o.risk_level)}</td><td class="num">${pct(o.probability, 1)}</td><td>${esc(o.seller_state)} → ${esc(o.customer_state)}</td></tr>`).join("") || `<tr><td colspan="4" class="skeleton">No alerts right now.</td></tr>`}</tbody></table></div>
        <a class="btn" style="margin-top:10px" href="#/queue?category=${key}&flagged=true">All alerts for ${esc(c.label)}</a></section>
    </div>
    <div class="callout outcome"><div class="k">Roadmap</div>${esc(d.roadmap_note)}</div>`;
  $$("[data-k]").forEach(b => b.addEventListener("click", () => go("focus", new URLSearchParams({ k: b.dataset.k }))));
  $$("tr[data-id]").forEach(tr => tr.addEventListener("click", () => openOrder(tr.dataset.id)));
  $$(".checklist input").forEach((cb, i) => cb.addEventListener("change", () => { checks[i] = cb.checked; store.set("ontime-playbook-" + key, checks); }));
  $("#f-copy").addEventListener("click", () => copyText($("#f-story").innerText));
  $("#f-rw").addEventListener("click", () => rewriteInto($("#f-story"), "executive", $("#f-eng"), $("#f-rw")));
  chart($("#f-month"), { type: "line", data: { labels: st.monthly.map(m => m.month), datasets: [
    { label: c.label, data: st.monthly.map(m => m.rate), borderColor: css("--accent"), backgroundColor: css("--accent"), tension: .25, pointRadius: 2 },
    { label: "Platform", data: st.monthly.map(() => st.platform_rate), borderColor: css("--faint"), borderDash: [4, 4], pointRadius: 0 }] },
    options: { plugins: { legend: { display: true, labels: { boxWidth: 12 } } }, scales: { y: { beginAtZero: true, ticks: { callback: v => v + "%" } }, x: { grid: { display: false } } } } });
  hbar($("#f-pat"), st.pattern_mix.map(m => `P${m.pattern} ${d.patterns[m.pattern] || ""}`.slice(0, 44)), st.pattern_mix.map(m => m.share), css("--chart-1"));
  hbar($("#f-st"), st.worst_states.map(s => s.key), st.worst_states.map(s => s.rate), css("--accent"), { tip: c2 => `${c2.raw}% late · ${fmt(st.worst_states[c2.dataIndex].n)} orders` });
}

// ------------------------------------------------------------------ NETWORK (simulated)
async function renderNetwork() {
  const n = await api("/api/network");
  const sens = n.sensitivity;
  const tip = (arr) => c => `${c.raw}% late · ${fmt(arr[c.dataIndex].n)} orders`;
  view.innerHTML = `
    <div class="banner"><b>Data note</b><span>Olist doesn't publish carriers, hubs or shipping modes. As the mentor suggested, these fields are simulated
      from real ones (states, distance, promise, freight, weight) with documented rules, so the delivery-model discussion has something concrete to work with.
      They are clearly labelled and are <b>not</b> used by the risk model.</span></div>
    <div class="grid g2">
      <section class="panel"><h2>Delivery model</h2><p class="hint">Direct if same state and under 150 km; regional hub inside a region; national hub-and-spoke otherwise.</p><div class="chart sm"><canvas id="n-model"></canvas></div></section>
      <section class="panel"><h2>Hub transfers</h2><p class="hint">0 direct, 1 regional, 2 national, plus 1 for long-haul into the North / North-East.</p><div class="chart sm"><canvas id="n-tr"></canvas></div></section>
      <section class="panel"><h2>Shipping mode</h2><p class="hint">Express for short promises or high freight per kg; Economy for long promises and cheap freight.</p><div class="chart sm"><canvas id="n-mode"></canvas></div></section>
      <section class="panel"><h2>Last-mile hub</h2><p class="hint">One hub per region, in a real logistics city.</p><div class="chart sm"><canvas id="n-hub"></canvas></div></section>
    </div>
    <section class="panel"><h2>Riskiest lanes</h2><p class="hint">Seller region to customer region, at least 100 orders.</p>
      <div class="table-wrap"><table><thead><tr><th>From</th><th>To</th><th>Hubs</th><th class="num">Orders</th><th class="num">Late</th></tr></thead>
      <tbody>${n.lanes.slice(0, 12).map(l => `<tr><td>${esc(l.from)}</td><td>${esc(l.to)}</td><td class="muted">${esc(l.hubs)}</td><td class="num">${fmt(l.n)}</td><td class="num"><b style="color:${l.rate > 10 ? "var(--crit)" : "inherit"}">${l.rate}%</b></td></tr>`).join("")}</tbody></table></div></section>
    <section class="panel"><h2>Do the logistics fields improve the model?</h2>
      <div class="mini-kpis"><div><b>${sens["without simulated fields"].toFixed(3)}</b><span>ROC-AUC without</span></div><div><b>${sens["with simulated fields"].toFixed(3)}</b><span>ROC-AUC with</span></div>
        <div><b>${sens.lift >= 0 ? "+" : ""}${sens.lift.toFixed(3)}</b><span>difference</span></div></div>
      <p class="note">No meaningful lift, which is exactly what should happen: the simulated fields are built from information the model already has. They help explain
        and plan (which lanes need a regional hub, where hub-and-spoke adds transfers), but only real carrier and route data could make the model more accurate.</p></section>`;
  vbar($("#n-model"), n.delivery_model.map(x => x.key), n.delivery_model.map(x => x.rate), css("--chart-1"), { tip: tip(n.delivery_model) });
  vbar($("#n-tr"), n.transfers.map(x => x.key + " transfer" + (x.key === "1" ? "" : "s")), n.transfers.map(x => x.rate), css("--accent"), { tip: tip(n.transfers) });
  vbar($("#n-mode"), n.shipping_mode.map(x => x.key), n.shipping_mode.map(x => x.rate), css("--chart-2"), { tip: tip(n.shipping_mode) });
  hbar($("#n-hub"), n.last_mile_hub.map(x => x.key), n.last_mile_hub.map(x => x.rate), css("--chart-1"), { tip: tip(n.last_mile_hub) });
}

// ------------------------------------------------------------------ BRIEFING
async function renderBriefing(params) {
  const ps = (await api("/api/briefing/periods")).periods;
  const period = params.get("period") || ps[0].key;
  const b = await api("/api/briefing?period=" + encodeURIComponent(period));
  view.innerHTML = `
    <div class="filters no-print" style="justify-content:space-between">
      <div class="field"><label for="b-period">Period</label><select id="b-period">${ps.map(p => `<option value="${p.key}" ${p.key === period ? "selected" : ""}>${esc(p.label)}</option>`).join("")}</select></div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-self:flex-end"><button class="btn" id="b-copy">Copy</button><button class="btn" id="b-md">Download .md</button>
        <button class="btn" id="b-print">Print / PDF</button><button class="btn primary" id="b-rw">Rewrite for leadership</button></div>
    </div>
    <section class="panel"><div class="panel-head"><div><h2>${esc(b.period_label)} briefing</h2><p class="hint">${esc(b.note)}</p></div><span id="b-eng"></span></div>
      <div class="story" id="b-story">${b.story.map(x => `<p>${esc(x)}</p>`).join("")}</div></section>
    <div class="tiles">
      <div class="tile"><span class="k">Orders scored</span><span class="v">${fmt(b.n_orders)}</span><span class="s">${b.prev ? `${fmt(b.prev.n_orders)} in ${esc(b.prev.period_label)}` : ""}</span></div>
      <div class="tile"><span class="k">Alerts</span><span class="v">${fmt(b.n_flagged)}</span><span class="s">${b.flagged_share}% of orders</span></div>
      <div class="tile"><span class="k">Critical</span><span class="v" style="color:var(--crit)">${fmt(b.n_critical)}</span><span class="s">risk 80%+</span></div>
      <div class="tile hl"><span class="k">Late orders caught</span><span class="v">${b.outcome.caught}%</span><span class="s">${b.outcome.flagged_late_rate}% of alerts were really late</span></div>
    </div>
    <div class="grid g2">
      <section class="panel"><h2>Patterns behind the alerts</h2><ol class="recs">${b.top_patterns.map(p => `<li><span><a class="pchip" href="#/patterns?p=${p.rank}">P${p.rank}</a> <b>${esc(p.name)}</b> · ${p.n} alerts<br><span class="muted">${esc(p.recommendation)}</span></span></li>`).join("")}</ol></section>
      <section class="panel"><h2>Focus categories this period</h2><div class="table-wrap"><table><thead><tr><th>Category</th><th class="num">Orders</th><th class="num">Alerts</th><th class="num">Alert rate</th></tr></thead>
        <tbody>${b.focus.map(f => `<tr><td><a href="#/focus?k=${f.key}">${esc(f.label)}</a> <span class="tier ${esc(f.tier.replace(/ /g, "-"))}">${esc(f.tier)}</span></td><td class="num">${fmt(f.n)}</td><td class="num">${fmt(f.alerts)}</td><td class="num">${f.flagged_share}%</td></tr>`).join("")}</tbody></table></div></section>
    </div>
    <section class="panel"><h2>Critical orders</h2><div class="table-wrap"><table><thead><tr><th>Order</th><th class="num">Risk</th><th>Category</th><th>Route</th><th>Pattern</th><th>Suggested action</th></tr></thead>
      <tbody>${b.critical_orders.map(o => `<tr class="clickable" data-id="${esc(o.order_id)}"><td class="mono">${esc(shortId(o.order_id))}</td><td class="num">${pct(o.probability, 1)}</td><td>${esc(nice(o.product_category))}</td><td>${esc(o.seller_state)} → ${esc(o.customer_state)}</td><td><span class="pchip">P${o.pattern}</span></td><td class="act">${action(o.suggested_action)}</td></tr>`).join("") || `<tr><td colspan="6" class="skeleton">No critical orders this period.</td></tr>`}</tbody></table></div></section>`;
  $("#b-period").addEventListener("change", e => go("briefing", new URLSearchParams({ period: e.target.value })));
  $$("tr[data-id]").forEach(tr => tr.addEventListener("click", () => openOrder(tr.dataset.id)));
  $("#b-copy").addEventListener("click", () => copyText($("#b-story").innerText));
  $("#b-print").addEventListener("click", () => window.print());
  $("#b-rw").addEventListener("click", () => rewriteInto($("#b-story"), "executive", $("#b-eng"), $("#b-rw")));
  $("#b-md").addEventListener("click", () => {
    const md = [`# OnTime briefing: ${b.period_label}`, "", ...$("#b-story").innerText.split("\n").filter(Boolean).map(x => x + "\n"),
      "## Patterns behind the alerts", ...b.top_patterns.map(p => `- **P${p.rank} ${p.name}** (${p.n} alerts): ${p.recommendation}`), "",
      "## Focus categories", ...b.focus.map(f => `- ${f.label}: ${f.alerts} alerts of ${f.n} orders (${f.flagged_share}%)`), "", `_${b.note}_`].join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
    a.download = `OnTime_briefing_${b.period}.md`; document.body.appendChild(a); a.click(); a.remove();
  });
}

// ------------------------------------------------------------------ ALERT PLANNER
async function renderPlanner() {
  const inp = store.get("ontime-planner", { cost_check: 15, cost_late: 120, effectiveness: 50, threshold: null });
  view.innerHTML = `
    <p class="lede">Every alert costs a coordinator's time; every missed late order costs a refund, a support call and maybe a customer. Set your own costs and see
      which alert threshold pays off. Numbers come from the ${S.meta ? fmt(S.meta.n_queue) : ""} newest orders, where the real outcome is known.</p>
    <section class="panel"><div class="formrow">
      <div class="field"><label for="pl-check">Cost to check one alert (R$)</label><input type="number" id="pl-check" min="0" step="1" value="${inp.cost_check}"><span class="help">Coordinator time, a phone call</span></div>
      <div class="field"><label for="pl-late">Cost of one late delivery (R$)</label><input type="number" id="pl-late" min="0" step="5" value="${inp.cost_late}"><span class="help">Refund, support, lost repeat business</span></div>
      <div class="field"><label for="pl-eff">Late orders a check rescues (%)</label><input type="number" id="pl-eff" min="0" max="100" step="5" value="${inp.effectiveness}"><span class="help">Not every warned order can be saved</span></div>
      <div class="field" style="grid-column:1/-1"><label for="pl-t">Alert threshold</label><div class="range-row"><input type="range" id="pl-t" min="0.05" max="0.95" step="0.01"><output id="pl-t-o"></output></div></div>
    </div></section>
    <div class="tiles" id="pl-tiles"></div>
    <div class="grid g2">
      <section class="panel"><h2>Coverage vs workload</h2><p class="hint">How many orders get an alert, and what share of late orders they catch.</p><div class="chart"><canvas id="pl-c1"></canvas></div></section>
      <section class="panel"><h2>Net saving</h2><p class="hint">Rescued late orders minus the cost of checking, at each threshold.</p><div class="chart"><canvas id="pl-c2"></canvas></div></section>
    </div>
    <div class="callout" id="pl-sum"></div>`;
  let data = null, t2;
  const draw = () => {
    const t = +$("#pl-t").value; $("#pl-t-o").textContent = t.toFixed(2);
    const row = data.rows.reduce((a, r) => Math.abs(r.threshold - t) < Math.abs(a.threshold - t) ? r : a);
    $("#pl-tiles").innerHTML = `
      <div class="tile"><span class="k">Alerts</span><span class="v">${fmt(row.alerts)}</span><span class="s">${row.alert_share}% of orders</span></div>
      <div class="tile"><span class="k">Late orders caught</span><span class="v">${row.recall}%</span><span class="s">${fmt(row.late_caught)} of ${fmt(data.n_late)}</span></div>
      <div class="tile"><span class="k">Alerts really late</span><span class="v">${row.precision ?? "—"}%</span><span class="s">base rate ${(data.n_late / data.n_orders * 100).toFixed(1)}%</span></div>
      <div class="tile hl"><span class="k">Net saving</span><span class="v">R$ ${fmt(row.net_saving)}</span><span class="s">on this sample</span></div>`;
    const b = data.best, cur = data.current;
    $("#pl-sum").innerHTML = `<div class="k">Recommendation</div>With these costs the best threshold is <b>${b.threshold.toFixed(2)}</b>: ${fmt(b.alerts)} alerts (${b.alert_share}%),
      catching ${b.recall}% of late orders, net saving R$ ${fmt(b.net_saving)}. The model's current threshold (${cur.threshold.toFixed(2)}, tuned for F1) gives R$ ${fmt(cur.net_saving)}.
      ${b.net_saving <= 0 ? "At these costs no threshold pays off: checking costs more than it saves." : ""}`;
    store.set("ontime-planner", { cost_check: +$("#pl-check").value, cost_late: +$("#pl-late").value, effectiveness: +$("#pl-eff").value, threshold: t });
    charts.forEach(c => c.destroy()); charts = [];
    const lab = data.rows.map(r => r.threshold.toFixed(2));
    const mark = (arr) => arr.map((v, i) => Math.abs(data.rows[i].threshold - t) < 0.005 ? 5 : 0);
    chart($("#pl-c1"), { type: "line", data: { labels: lab, datasets: [
      { label: "Late orders caught %", data: data.rows.map(r => r.recall), borderColor: css("--accent"), pointRadius: mark(data.rows), pointBackgroundColor: css("--accent") },
      { label: "Orders alerted %", data: data.rows.map(r => r.alert_share), borderColor: css("--chart-1"), pointRadius: mark(data.rows) },
      { label: "Alerts really late %", data: data.rows.map(r => r.precision), borderColor: css("--faint"), borderDash: [4, 4], pointRadius: 0 }] },
      options: { plugins: { legend: { display: true, labels: { boxWidth: 12 } } }, scales: { y: { beginAtZero: true, max: 100, ticks: { callback: v => v + "%" } }, x: { ticks: { maxTicksLimit: 10 }, grid: { display: false } } } } });
    chart($("#pl-c2"), { type: "line", data: { labels: lab, datasets: [{ data: data.rows.map(r => r.net_saving), borderColor: css("--low"), backgroundColor: css("--low-soft"), fill: true, pointRadius: mark(data.rows), pointBackgroundColor: css("--low") }] },
      options: { scales: { y: { ticks: { callback: v => "R$ " + fmt(v) } }, x: { ticks: { maxTicksLimit: 10 }, grid: { display: false } } } } });
  };
  const load = async () => {
    const q = new URLSearchParams({ cost_check: $("#pl-check").value || 0, cost_late: $("#pl-late").value || 0, effectiveness: (+$("#pl-eff").value || 0) / 100 });
    data = await api("/api/planner?" + q);
    if (inp.threshold == null) { inp.threshold = data.current.threshold; }
    if (!$("#pl-t").dataset.init) { $("#pl-t").value = inp.threshold; $("#pl-t").dataset.init = "1"; }
    draw();
  };
  ["#pl-check", "#pl-late", "#pl-eff"].forEach(id => $(id).addEventListener("input", () => { clearTimeout(t2); t2 = setTimeout(load, 300); }));
  $("#pl-t").addEventListener("input", draw);
  await load();
}

// ------------------------------------------------------------------ ASSISTANT
S.chat = S.chat || [];
async function renderAssistant() {
  const [st, db] = await Promise.all([api("/api/assistant/status"), api("/api/database/status")]);
  if (!S.chatEngine) S.chatEngine = st.settings.engine;
  const eng = st.engines;
  const names = { builtin: "Built-in", laptop: "Laptop model", gemini: "Cloud AI", auto: "Auto" };
  view.innerHTML = `
    <div class="chat">
      <div class="chat-head">
        <div class="seg" role="group" aria-label="Engine">${["auto", "builtin", "laptop", "gemini"].map(k => `<button type="button" data-eng="${k}" aria-pressed="${S.chatEngine === k}"
          title="${esc(eng[k].detail)}"><span class="dot ${eng[k].available ? "on" : "off"}"></span>${names[k]}</button>`).join("")}</div>
        <button class="btn sm ghost" type="button" id="chat-clear" title="Start a new conversation">New chat</button>
        <span class="muted" style="font-size:13px">Time limit ${st.settings.timeout_s} s · falls back to built-in · <a href="#/settings">Settings</a></span>
      </div>
      <div class="chat-log" id="chat-log"></div>
      <form class="chat-input" id="chat-form"><label class="sr" for="chat-q">Your question</label>
        <textarea id="chat-q" rows="1" placeholder="Ask about an order, a pattern, a state, a category or the model…"></textarea>
        <button class="btn primary" type="submit" id="chat-send">Ask</button></form>
    </div>`;
  const log = $("#chat-log");
  const drawLog = () => {
    if (!S.chat.length) {
      log.innerHTML = `<div class="welcome"><h3>Ask OnTime</h3><p>Answers come from the app's own numbers. The built-in engine works offline; the laptop and cloud AI engines
        reword the same facts and never add numbers of their own.</p><div class="chips">${st.examples.map(e => `<button class="chip" type="button" data-ex="${esc(e)}">${esc(e)}</button>`).join("")}</div></div>`;
    } else {
      log.innerHTML = S.chat.map(m => m.role === "user" ? `<div class="msg user">${esc(m.text)}</div>` : m.pending
        ? `<div class="msg bot"><span class="typing"><i></i><i></i><i></i></span><span class="muted" style="font-size:12.5px" id="chat-timer">Thinking…</span></div>`
        : `<div class="msg bot">${m.r.paragraphs.map(p => `<p>${esc(p)}</p>`).join("")}
          ${m.r.resolved_as ? `<div class="note">Read as a question about ${esc(m.r.resolved_as)}.</div>` : ""}
          ${m.r.fallback_reason && m.r.requested_engine !== "builtin" ? `<div class="note">${esc(names[m.r.requested_engine])} unavailable: ${esc(m.r.fallback_reason)}</div>` : ""}
          <div class="foot">${engineTag(m.r.engine, m.r.elapsed_ms, m.r.fallback_reason && m.r.requested_engine !== "builtin" ? m.r.fallback_reason : null)}
            ${m.r.sources.map(s => `<span class="tag">${esc(s)}</span>`).join("")}
            ${m.r.order_id ? `<button class="btn sm" data-order="${esc(m.r.order_id)}">Open order</button>` : ""}
            <button class="btn sm ghost" data-copy="${S.chat.indexOf(m)}">Copy</button></div>
          ${m.r.follow_ups && m.r.follow_ups.length ? `<div class="follow">${m.r.follow_ups.map(f => `<button class="chip" type="button" data-ex="${esc(f)}">${esc(f)}</button>`).join("")}</div>` : ""}</div>`).join("");
    }
    log.scrollTop = log.scrollHeight;
    $$("[data-ex]", log).forEach(b => b.addEventListener("click", () => send(b.dataset.ex)));
    $$("[data-order]", log).forEach(b => b.addEventListener("click", () => openOrder(b.dataset.order)));
    $$("[data-copy]", log).forEach(b => b.addEventListener("click", () => { const m = S.chat[+b.dataset.copy]; if (m && m.r) copyText(m.r.paragraphs.join("\n\n")); }));
  };
  const send = async (q) => {
    q = (q || "").trim(); if (!q) return;
    S.chat.push({ role: "user", text: q }); const pend = { role: "bot", pending: true }; S.chat.push(pend); drawLog();
    $("#chat-q").value = ""; $("#chat-send").disabled = true;
    const t0 = Date.now(); const tick = setInterval(() => { const el = $("#chat-timer"); if (el) el.textContent = `Thinking… ${((Date.now() - t0) / 1000).toFixed(0)} s`; }, 500);
    const hist = S.chat.filter(m => !m.pending).slice(-6).map(m => ({ role: m.role === "user" ? "user" : "assistant", text: m.role === "user" ? m.text : (m.r ? m.r.paragraphs.join(" ") : "") }));
    try { pend.r = await api("/api/assistant/ask", { method: "POST", body: JSON.stringify({ question: q, engine: S.chatEngine, context: S.chatCtx || null, history: hist.slice(0, -1) }) });
      S.chatCtx = (pend.r.context && Object.keys(pend.r.context).length) ? pend.r.context : S.chatCtx; }
    catch (e) { pend.r = { paragraphs: [e.message], sources: [], engine: "builtin", requested_engine: S.chatEngine, follow_ups: [] }; }
    clearInterval(tick); pend.pending = false; $("#chat-send").disabled = false; drawLog(); $("#chat-q").focus();
  };
  $$("[data-eng]").forEach(b => b.addEventListener("click", () => { S.chatEngine = b.dataset.eng; $$("[data-eng]").forEach(x => x.setAttribute("aria-pressed", x === b)); toast("Switched to " + names[b.dataset.eng] + " engine"); const head = $(".chat-head"); if (head) { head.classList.add("engine-switched"); setTimeout(() => head.classList.remove("engine-switched"), 400); } }));
  $("#chat-form").addEventListener("submit", e => { e.preventDefault(); send($("#chat-q").value); });
  $("#chat-q").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($("#chat-q").value); } });
  $("#chat-clear").addEventListener("click", () => { S.chat = []; S.chatCtx = null; drawLog(); });
  drawLog();
}

// ------------------------------------------------------------------ SETTINGS
async function renderSettings() {
  const [st, db] = await Promise.all([api("/api/assistant/status"), api("/api/database/status")]);
  const s = st.settings, eng = st.engines;
  view.innerHTML = `
    <section class="panel"><h2>Appearance</h2><p class="hint">Eight themes. The button at the bottom of the menu switches between the light and dark version of the same colour.</p>
      ${swatchesHtml()}
      <div class="chips2" style="margin-top:12px">${THEMES.map(t => `<span class="dchip">${t.label} · ${t.mode}</span>`).join("")}</div></section>
    <section class="panel" id="set-db-panel"><h2>Database storage</h2>
      <p class="hint">Choose where the action board and team activity logs are stored. SQLite runs locally with zero setup; Supabase Cloud stores data in hosted PostgreSQL so a team can share the queue.</p>
      <div class="radio-cards">
        <label class="radio-card"><input type="radio" name="db_mode" value="sqlite" ${(!s.database_mode || s.database_mode === "sqlite") ? "checked" : ""}>
          <b>Local SQLite <span class="dot on"></span></b><span>Offline-first, zero setup, runs on your computer.</span></label>
        <label class="radio-card"><input type="radio" name="db_mode" value="supabase" ${s.database_mode === "supabase" ? "checked" : ""}>
          <b>Supabase Cloud <span class="dot ${db.connected ? "on" : "off"}"></span></b><span>PostgreSQL cloud storage. Multi-user coordination across teams.</span></label>
      </div>
      <div id="sb-fields" style="margin-top:14px;${s.database_mode === "supabase" ? "" : "display:none"}">
        <div class="formrow"><div class="field"><label for="set-sb-url">Supabase Project URL</label><input id="set-sb-url" placeholder="https://yourproject.supabase.co" value="${esc(s.supabase_url || "")}"></div>
          <div class="field"><label for="set-sb-key">Supabase API Key (Anon or Service Role)</label><input id="set-sb-key" type="password" autocomplete="off" placeholder="${s.supabase_key_set ? "Saved (" + esc(s.supabase_key_hint) + ") – type to replace" : "Paste your key"}"></div></div>
        <p class="note">Get credentials at <a href="https://supabase.com" target="_blank" rel="noopener">supabase.com</a>. Saved to <code>backend/settings.json</code> and your project <code>.env</code> automatically.</p>
        <div class="story-tools" style="margin-top:8px">
          <button class="btn sm" id="t-supabase" type="button">Test Supabase connection</button>
          <button class="btn sm ghost" id="btn-copy-sql" type="button">Copy SQL setup schema</button>
          <span id="t-supabase-r" class="muted"></span>
        </div>
      </div>
    </section>
    <section class="panel"><h2>Scoring model</h2><p class="hint">The model chosen here is used everywhere: Overview, Risk queue, Action board alerts, delay patterns, briefings, Model page, Score an order and Batch scoring. Orders, bands, alerts and next actions are re-scored with it the moment you choose. The dispatch model needs dispatch-time signals the historical orders do not have, so it scores single orders and batches while the queue stays on the real-Olist model.</p>
      <div class="radio-cards">${[["auto", "Automatic", "The default: the enriched chronological model (" + (S.meta.base_serving === "enriched" ? "real Olist orders, chronological test" : S.meta.base_serving === "ensemble" ? "weighted ensemble" : "tuned XGBoost") + ")."],
        ["xgboost", "Tuned XGBoost", "The original production model. Trained on a random split, so queue figures are optimistic."],
        ["ensemble", "Weighted ensemble", S.meta.ensemble_available ? "XGBoost, histogram boosting and logistic regression, blended. Trained on a random split, so queue figures are optimistic." : "Not available: run train_ensemble.py first."],
        ["enriched", "Enriched chronological model", S.meta.enriched_available ? "70 checkout-time features; real Olist data, chronological validation." : "Not available: train the enriched model first."],
        ["synthetic_operational", "Dispatch operational model", S.meta.synthetic_operational_available ? "Synthetic benchmark only; opt in for demonstrations. Scores single orders and batches; the queue stays on the real-Olist model." : "Not available: model artifact missing."]].map(([k, t, d]) => `
        <label class="radio-card"><input type="radio" name="scoring" value="${k}" ${s.scoring_model === k ? "checked" : ""} ${k === "ensemble" && !S.meta.ensemble_available || k === "enriched" && !S.meta.enriched_available || k === "synthetic_operational" && !S.meta.synthetic_operational_available ? "disabled" : ""}><b>${t}</b><span>${d}</span></label>`).join("")}</div></section>
    <div class="settings-grid">
      <section class="panel"><h2>Assistant engine</h2><p class="hint">Used by Ask OnTime and the "Rewrite" buttons. If the chosen engine fails or runs out of time, the built-in answer is used.</p>
        <form id="set-form">
        <div class="radio-cards">
          ${[["builtin", "Built-in", "Offline, instant, always available. Answers from the app's numbers with fixed wording."],
             ["auto", "Auto (recommended)", "Uses the cloud AI if a key is set, else the laptop model if Ollama is running, else the built-in engine. Falls back by itself if one fails."],
             ["laptop", "Laptop model (Ollama)", "A local AI model on this computer. Private: nothing leaves the machine. Needs Ollama installed."],
             ["gemini", "Cloud AI", "A hosted AI model, used with your own API key. Best wording; needs internet."]].map(([k, t, d]) => `
            <label class="radio-card"><input type="radio" name="engine" value="${k}" ${s.engine === k ? "checked" : ""}><b>${t} <span class="dot ${eng[k].available ? "on" : "off"}"></span></b>
              <span>${d}<br><i>${esc(eng[k].detail)}</i></span></label>`).join("")}
        </div>
        <div class="field" style="margin-top:14px"><label for="set-timeout">Time limit for AI engines</label>
          <div class="range-row"><input type="range" id="set-timeout" min="5" max="${s.max_timeout_s}" step="5" value="${s.timeout_s}"><output id="set-timeout-o">${s.timeout_s} s</output></div>
          <span class="help">5 seconds to 2 minutes. After this the built-in answer is shown instead.</span></div>
        <div style="margin-top:14px;display:flex;gap:8px"><button class="btn primary" type="submit">Save settings</button></div>
        </form></section>
      <section class="panel"><h2>Engine details</h2>
        <h4 ${H4S}>Laptop model (Ollama)</h4>
        <div class="formrow"><div class="field"><label for="set-ourl">Ollama address</label><input id="set-ourl" value="${esc(s.ollama_url)}"></div>
          <div class="field"><label for="set-omodel">Model</label><input id="set-omodel" list="omodels" value="${esc(s.ollama_model)}">
            <datalist id="omodels">${(eng.laptop.models || []).map(m => `<option value="${esc(m)}">`).join("")}</datalist></div></div>
        <p class="note">Install from ollama.com, then run <span class="mono">ollama pull ${esc(s.ollama_model)}</span>. Small models (llama3.2, qwen2.5:3b) run on most laptops.</p>
        <div class="story-tools"><button class="btn sm" id="t-laptop" type="button">Test laptop engine</button><span id="t-laptop-r" class="muted"></span></div>
        <h4 style="margin:18px 0 10px;font:600 17px var(--display);letter-spacing:.02em">Cloud AI engine</h4>
        <div class="formrow"><div class="field"><label for="set-gmodel">Model</label><input id="set-gmodel" value="${esc(s.gemini_model)}"></div>
          <div class="field"><label for="set-gkey">API key</label><input id="set-gkey" type="password" autocomplete="off" placeholder="${s.gemini_key_set ? "Saved (" + esc(s.gemini_key_hint) + ") – type to replace" : "Paste your key"}"></div></div>
        <p class="note">Get a free key at aistudio.google.com. It is saved only on this computer (backend/settings.json) or read from GEMINI_API_KEY in .env.</p>
        <div class="story-tools"><button class="btn sm" id="t-gemini" type="button">Test cloud AI</button><span id="t-gemini-r" class="muted"></span></div></section>
    </div>`;
  bindSwatches(view);
  $("#set-timeout").addEventListener("input", e => $("#set-timeout-o").textContent = e.target.value + " s");
  const save = async (quiet) => {
    const dbMode = $("input[name=db_mode]:checked") ? $("input[name=db_mode]:checked").value : "sqlite";
    const body = { engine: $("input[name=engine]:checked").value, timeout_s: +$("#set-timeout").value, database_mode: dbMode, ollama_url: $("#set-ourl").value.trim(),
      ollama_model: $("#set-omodel").value.trim(), gemini_model: $("#set-gmodel").value.trim(),
      scoring_model: ($("input[name=scoring]:checked") || { value: "auto" }).value };
    if ($("#set-gkey").value.trim()) body.gemini_api_key = $("#set-gkey").value.trim();
    if ($("#set-sb-url")) body.supabase_url = $("#set-sb-url").value.trim();
    if ($("#set-sb-key") && $("#set-sb-key").value.trim()) body.supabase_key = $("#set-sb-key").value.trim();
    await api("/api/settings", { method: "POST", body: JSON.stringify(body) });
    S.chatEngine = body.engine;
    if (!quiet) { toast("Settings saved"); renderSettings(); }
  };
  // Changing the scoring model or assistant engine takes effect straight away; no need to hunt for the Save button.
  const autosave = async () => { try { await save(true); await refreshAfterModelSwitch(); const v = S.meta.queue_view || {}; toast("Saved. Every page now uses " + (v.label || "the selected model") + (v.limited_by ? " for the queue (the dispatch model only scores single orders and batches)" : "")); if (parseHash().page === "settings") renderSettings(); } catch (er) { toast(er.message); } };
  $$("input[name=scoring], input[name=engine]").forEach(i => i.addEventListener("change", autosave));
  $("#set-form").addEventListener("submit", async e => { e.preventDefault(); try { await save(); } catch (er) { toast(er.message); } });
  const test = async (engine) => {
    const out = $(`#t-${engine}-r`); out.textContent = "Testing…";
    try { await save(true); const r = await api(`/api/assistant/test?engine=${engine}`, { method: "POST" });
      out.innerHTML = r.ok ? `<span class="ontime">Working · ${(r.elapsed_ms / 1000).toFixed(1)} s</span>` : `<span class="err">${esc(r.error)}</span>`;
    } catch (e) { out.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  };
  $("#t-laptop").addEventListener("click", () => test("laptop"));
  $("#t-gemini").addEventListener("click", () => test("gemini"));
  $$("input[name=db_mode]").forEach(r => r.addEventListener("change", e => {
    const sb = $("#sb-fields"); if (sb) sb.style.display = e.target.value === "supabase" ? "" : "none";
  }));
  const testSb = async () => {
    const out = $("#t-supabase-r"); out.textContent = "Testing...";
    try {
      const url = $("#set-sb-url").value.trim(), key = $("#set-sb-key").value.trim();
      const r = await api("/api/database/test", { method: "POST", body: JSON.stringify({ url: url || null, key: key || null }) });
      out.innerHTML = r.ok ? `<span class="ontime">Connected · ${r.latency_ms} ms</span>` : `<span class="err">${esc(r.detail)}</span>`;
      updateDbStatus();
    } catch (e) { out.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  };
  const tSb = $("#t-supabase"); if (tSb) tSb.addEventListener("click", testSb);
  const btnSql = $("#btn-copy-sql");
  if (btnSql) btnSql.addEventListener("click", () => copyText(db.schema_sql));
}

// ------------------------------------------------------------------ LITERATURE
async function renderLiterature() {
  const L = await api("/api/literature");
  const total = L.themes.reduce((n, t) => n + t.papers.length, 0);
  const card = (p) => `<article class="lit-card" data-text="${esc((p.title + " " + p.authors + " " + p.what + " " + p.used + " " + p.venue).toLowerCase())}">
    <h4>${esc(p.title)}</h4><div class="lit-meta">${esc(p.authors)} · ${esc(p.year)} · <i>${esc(p.venue)}</i></div>
    <p><b>What it found.</b> ${esc(p.what)}</p><p><b>In OnTime.</b> ${esc(p.used)}</p>
    ${p.link ? `<a class="btn sm" href="${esc(p.link)}" target="_blank" rel="noopener">Open source ↗</a>` : `<span class="muted" style="font-size:12.5px">No stable link: look it up by title and venue.</span>`}</article>`;
  view.innerHTML = `
    <p class="lede">${esc(L.intro)}</p>
    <div class="tiles"><div class="tile hl"><span class="k">Sources</span><span class="v">${total}</span><span class="s">in ${L.themes.length} themes</span></div>
      <div class="tile"><span class="k">Late-delivery studies</span><span class="v">${L.themes[0].papers.length}</span><span class="s">applied work, 2020 to 2025</span></div>
      <div class="tile"><span class="k">Last updated</span><span class="v" style="font-size:22px">${esc(L.updated)}</span><span class="s">domain papers checked against their abstract page</span></div></div>
    <section class="panel"><h2>Where OnTime sits</h2><ul class="steps">${L.positioning.map((t, i) => `<li><b>${i + 1}</b><span>${esc(t)}</span></li>`).join("")}</ul></section>
    <section class="panel"><div class="field" style="max-width:420px"><label for="lit-q">Filter the sources</label><input id="lit-q" placeholder="e.g. XGBoost, leakage, SHAP, imbalance"></div>
      <p class="muted" id="lit-count" style="margin:8px 0 0;font-size:13px"></p></section>
    <div id="lit-themes">${L.themes.map(t => `<section class="panel lit-theme"><h2>${esc(t.title)}</h2>${t.summary ? `<p class="hint">${esc(t.summary)}</p>` : ""}<div class="lit-grid">${t.papers.map(card).join("")}</div></section>`).join("")}</div>
    <section class="panel"><h2>Limits of this review</h2><ul class="steps">${L.limits.map(t => `<li><span>${esc(t)}</span></li>`).join("")}</ul></section>`;
  const apply = () => {
    const q = $("#lit-q").value.trim().toLowerCase(); let n = 0;
    $$(".lit-card").forEach(c => { const ok = !q || c.dataset.text.includes(q); c.style.display = ok ? "" : "none"; if (ok) n++; });
    $$(".lit-theme").forEach(t => { t.style.display = $$(".lit-card", t).some(c => c.style.display !== "none") ? "" : "none"; });
    $("#lit-count").textContent = q ? `${n} of ${total} sources match.` : "";
  };
  $("#lit-q").addEventListener("input", apply);
}

// ------------------------------------------------------------------ ADAPT TO YOUR DATA
const PROV_LABEL = { mapped: "from your file", derived: "worked out from your file", assumed: "assumed", lookup: "past-rate lookup", default: "typical value" };
const PROV_CLASS = { mapped: "ontime", derived: "ontime", assumed: "warn-t", lookup: "ontime", default: "late" };

async function renderAdapt() {
  const A = (S.adapt = S.adapt || {});
  view.innerHTML = `
    <p class="lede">Your orders will not look like the Olist export: different column names, units and date formats, maybe no weight or distance at all. Upload a CSV and OnTime matches its columns to what the model needs, converts what it can, and says exactly which inputs came from your file and which had to be filled in. If your file also says which orders were late, it measures how well the Olist model transfers, and can learn a model from your own history.</p>
    <div class="grid g2">
      <label class="dropzone" id="ad-dz" for="ad-file"><b>Drop your CSV here</b><span class="muted">or click to choose a file (up to 20,000 rows)</span>
        <input type="file" id="ad-file" accept=".csv,text/csv" class="sr"><span class="btn primary">Choose file</span></label>
      <section class="panel"><h2>What helps most</h2>
        <ul class="steps"><li><b>Order date and promised date</b><span>give the promise window and the timing inputs.</span></li>
          <li><b>Where it goes</b><span>a state or region, and either a distance or coordinates.</span></li>
          <li><b>The outcome</b><span>delivered date, or a column saying late / on time, so results can be checked and a model learned.</span></li>
          <li><b>Seller and category</b><span>unlock the past-delay rates.</span></li></ul></section>
    </div>
    <div id="ad-map"></div><div id="ad-out"></div>
    <section class="panel" id="ad-models"></section>`;
  const dz = $("#ad-dz");
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) adaptUpload(f); });
  $("#ad-file").addEventListener("change", e => { const f = e.target.files[0]; if (f) adaptUpload(f); });
  adaptModelsPanel();
}

async function adaptUpload(file) {
  const box = $("#ad-map"); $("#ad-out").innerHTML = "";
  box.innerHTML = `<div class="panel skeleton">Reading ${esc(file.name)}…</div>`;
  const fd = new FormData(); fd.append("file", file);
  let p;
  try { p = await api("/api/adapt/preview", { method: "POST", body: fd }); }
  catch (e) { box.innerHTML = `<div class="callout bad"><div class="k">Could not read this file</div>${esc(e.message)}</div>`; return; }
  S.adapt.p = p;
  const cols = p.columns.map(c => c.name);
  const sample = Object.fromEntries(p.columns.map(c => [c.name, c.sample.join(" · ")]));
  const priority = ["purchase_time", "promised_time", "promised_days", "customer_state", "seller_state", "distance", "delivered_time", "late_flag"];
  const order = [...priority, ...p.concepts.map(c => c.key).filter(k => !priority.includes(k))];
  const byKey = Object.fromEntries(p.concepts.map(c => [c.key, c]));
  const row = (k) => {
    const c = byKey[k], sg = p.suggested[k] || {};
    return `<tr data-k="${k}"><td><b>${esc(c.label)}</b></td>
      <td><select class="ad-src" data-k="${k}"><option value="">not in my file</option>${cols.map(n => `<option value="${esc(n)}" ${sg.source === n ? "selected" : ""}>${esc(n)}</option>`).join("")}</select></td>
      <td>${c.units ? `<select class="ad-unit" data-k="${k}">${c.units.map(u => `<option ${sg.unit === u ? "selected" : ""}>${u}</option>`).join("")}</select>` : ""}
          ${c.bases ? `<select class="ad-basis" data-k="${k}" title="Is this value for the whole order or for one item?">${c.bases.map(b => `<option value="${b}" ${sg.basis === b ? "selected" : ""}>${b === "total" ? "whole order" : "per item"}</option>`).join("")}</select>` : ""}</td>
      <td class="muted ad-sample" data-k="${k}" style="font-size:12.5px">${sg.source ? esc(sample[sg.source]) : ""}</td>
      <td>${sg.source ? `<span class="tag">${esc(sg.how)}${sg.confidence < 0.7 ? " · check" : ""}</span>` : ""}</td></tr>`;
  };
  const o = p.outcome;
  box.innerHTML = `<section class="panel"><h2>1. Check the column matching</h2>
    <p class="hint">${esc(p.filename)} · ${fmt(p.n_rows)} rows · ${p.n_columns} columns. Suggestions are guessed from names and checked against the values (a date column must parse as dates). Correct anything that is wrong.</p>
    <div class="table-wrap"><table class="ad-table"><thead><tr><th>OnTime needs</th><th>Your column</th><th>Unit</th><th>Sample values</th><th></th></tr></thead><tbody id="ad-rows">${order.map(row).join("")}</tbody></table></div>
    <div class="callout ${o.found ? "outcome" : "warn"}" id="ad-outcome" style="margin-top:14px"></div>
    <div style="margin-top:14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap">
      <label class="toggle"><input type="checkbox" id="ad-train"> Also learn a model from my own history</label>
      <div class="field" style="min-width:200px"><label for="ad-name" class="sr">Name for the saved model</label><input id="ad-name" placeholder="Name this model (optional)" maxlength="60"></div>
      <button class="btn primary" id="ad-run">Check the data and score it</button></div></section>`;
  const mapping = () => {
    const m = {};
    $$(".ad-src").forEach(s => { if (s.value) { const k = s.dataset.k; m[k] = { source: s.value };
      const u = $(`.ad-unit[data-k="${k}"]`), b = $(`.ad-basis[data-k="${k}"]`); if (u) m[k].unit = u.value; if (b) m[k].basis = b.value; } });
    return m;
  };
  S.adapt.mapping = mapping;
  $$(".ad-src").forEach(s => s.addEventListener("change", () => { $(`.ad-sample[data-k="${s.dataset.k}"]`).textContent = s.value ? sample[s.value] : ""; }));
  const outcomeText = () => {
    const m = mapping();
    const has = m.late_flag || (m.delivered_time && (m.promised_time));
    $("#ad-outcome").className = "callout " + (has ? "outcome" : "warn");
    $("#ad-outcome").innerHTML = has ? `<div class="k">Outcome</div>Your file says which orders were late (${m.late_flag && !(m.delivered_time && m.promised_time) ? "from the “" + esc(m.late_flag.source) + "” column" : "delivered date after the promised date"}). OnTime can check its predictions against it${o.found && o.can_learn ? ` (${fmt(o.n_labelled)} labelled orders, ${pct(o.late_rate, 1)} late)` : ""}.`
      : `<div class="k">No outcome mapped</div>Without the delivered date (plus promised date) or a late flag, OnTime can score your orders but cannot tell you how accurate the scores are or learn from your history.`;
  };
  outcomeText();
  $$(".ad-src").forEach(s => s.addEventListener("change", outcomeText));
  $("#ad-run").addEventListener("click", () => adaptRun());
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function adaptRun() {
  const out = $("#ad-out"), btn = $("#ad-run");
  btn.disabled = true; out.innerHTML = `<div class="panel skeleton">Matching, scoring and checking… (${$("#ad-train").checked ? "training a model on your history takes up to a minute" : "a few seconds"})</div>`;
  let r;
  try { r = await api("/api/adapt/run", { method: "POST", body: JSON.stringify({ dataset_id: S.adapt.p.dataset_id, mapping: S.adapt.mapping(), train: $("#ad-train").checked, name: $("#ad-name").value.trim() || null }) }); }
  catch (e) { out.innerHTML = `<div class="callout bad"><div class="k">Could not run</div>${esc(e.message)}</div>`; btn.disabled = false; return; }
  btn.disabled = false;
  const cv = r.coverage, t = r.transfer, m = t.metrics, a = r.adapted;
  const prov = Object.entries(cv.provenance);
  const label = (f) => S.meta.feature_labels[f] || nice(f);
  const metricTiles = (x, hl) => x && x.roc_auc != null ? `<div class="tiles"><div class="tile ${hl ? "hl" : ""}"><span class="k">ROC-AUC</span><span class="v">${x.roc_auc.toFixed(3)}</span><span class="s">0.5 = guessing</span></div>
      <div class="tile"><span class="k">PR-AUC</span><span class="v">${x.pr_auc.toFixed(3)}</span><span class="s">base rate ${pct(x.late_rate, 1)}</span></div>
      <div class="tile"><span class="k">Precision</span><span class="v">${pct(x.precision)}</span><span class="s">of alerts really late</span></div>
      <div class="tile"><span class="k">Recall</span><span class="v">${pct(x.recall)}</span><span class="s">of late orders alerted</span></div>
      <div class="tile"><span class="k">Top 10% of orders</span><span class="v">${pct(x.recall_at_top10pct)}</span><span class="s">of late orders inside</span></div></div>` : (x ? `<p class="note">${esc(x.note || "")}</p>` : "");
  out.innerHTML = `
    <section class="panel"><h2>2. What reached the model</h2>
      <div class="tiles"><div class="tile hl"><span class="k">Inputs from your file</span><span class="v">${cv.n_real} / ${cv.n_features}</span><span class="s">mapped or worked out</span></div>
        <div class="tile"><span class="k">Model weight covered</span><span class="v">${pct(cv.importance_coverage)}</span><span class="s">share of the model's importance that rests on real inputs</span></div>
        <div class="tile"><span class="k">Rows</span><span class="v">${fmt(r.n_rows)}</span><span class="s">${esc(r.filename)}</span></div>
        <div class="tile"><span class="k">Outcome</span><span class="v" style="font-size:20px">${r.outcome.found ? pct(r.outcome.late_rate, 1) + " late" : "none"}</span><span class="s">${r.outcome.found ? esc(r.outcome.how) : "not in the file"}</span></div></div>
      ${cv.warning ? `<div class="callout warn"><div class="k">Low coverage</div>${esc(cv.warning)}</div>` : ""}
      ${cv.notes.length ? `<p class="note">${cv.notes.map(esc).join(" ")}</p>` : ""}
      <details style="margin-top:10px"><summary style="cursor:pointer;font-weight:600">Where every model input came from</summary>
        <div class="prov-grid">${prov.map(([f, how]) => `<div class="prov"><span>${esc(label(f))}</span><span class="tag ${PROV_CLASS[how] || ""}">${PROV_LABEL[how] || how}</span></div>`).join("")}</div></details></section>
    <section class="panel"><h2>3. The Olist model on your data <span class="tag">${esc(t.model_label)}</span></h2>
      <div class="callout ${t.verdict.level === "good" ? "outcome" : t.verdict.level === "weak" ? "bad" : "warn"}"><div class="k">${t.verdict.level === "unknown" ? "Not measured" : "Transfer check"}</div>${esc(t.verdict.text)}</div>
      ${metricTiles(m, true)}
      <p class="note">${fmt(t.scored)} of ${fmt(t.rows_scored_for_check)} rows checked were scored (${fmt(t.rejected)} rejected, ${fmt(t.low_confidence)} low confidence); ${t.flagged_share != null ? pct(t.flagged_share, 1) + " raised an alert" : ""}. Bands: ${BANDS.map(b => `${b.toLowerCase()} ${t.bands[b]}`).join(" · ")}.</p>
      ${t.reject_reasons.length ? `<div class="callout warn"><div class="k">Why rows were rejected</div>${t.reject_reasons.map(x => `${esc(x.reason)} (${fmt(x.n)})`).join("<br>")}</div>` : ""}</section>
    ${a ? `<section class="panel"><h2>4. A model learned from your own history <span class="tag">${esc(a.name || "")}</span></h2>
      <div class="callout outcome"><div class="k">Result</div>${esc(a.verdict)}</div>
      <div class="table-wrap"><table><thead><tr><th>On the same ${fmt(a.n_test)} ${a.split === "time" ? "most recent" : "random"} orders</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">Precision</th><th class="num">Recall</th></tr></thead><tbody>
        <tr style="background:var(--accent-soft);font-weight:700"><td>Learned from your data</td><td class="num">${a.roc_auc.toFixed(3)}</td><td class="num">${a.pr_auc.toFixed(3)}</td><td class="num">${pct(a.precision)}</td><td class="num">${pct(a.recall)}</td></tr>
        <tr><td>Olist model, as it is</td>${a.same_rows_olist_model && a.same_rows_olist_model.roc_auc != null ? `<td class="num">${a.same_rows_olist_model.roc_auc.toFixed(3)}</td><td class="num">${a.same_rows_olist_model.pr_auc.toFixed(3)}</td><td class="num">${pct(a.same_rows_olist_model.precision)}</td><td class="num">${pct(a.same_rows_olist_model.recall)}</td>` : `<td colspan="4" class="muted">not measurable</td>`}</tr></tbody></table></div>
      <p class="note">Trained on ${fmt(a.n_train)} orders (${a.split === "time" ? "oldest 80% by date; tested on the newest 20%" : "random 80%; no usable order date, so tested on a random 20%"}), ${a.own_history_rates.length ? "with past-delay rates computed from your own history, using only orders already delivered" : "without past-delay rates (they need order and delivery dates)"}. Alert threshold ${a.threshold} was chosen on training data only.</p>
      <h4 style="margin:14px 0 8px;font:600 17px var(--display)">What it relies on</h4>
      <div class="chart" style="height:${24 + a.importance.length * 26}px"><canvas id="ad-imp"></canvas></div>
      <p class="note">Saved as <span class="mono">${esc(a.model_id)}</span>. Use it below to score a new file with the same columns.</p></section>` : ""}`;
  if (a) hbar($("#ad-imp"), a.importance.map(x => label(x.feature)), a.importance.map(x => x.importance), css("--chart-1"), { unit: "", tip: c => c.raw.toFixed(4) });
  adaptModelsPanel();
  out.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function adaptModelsPanel() {
  const el = $("#ad-models"); if (!el) return;
  let r; try { r = await api("/api/adapt/models"); } catch (e) { el.innerHTML = ""; return; }
  if (!r.models.length) { el.innerHTML = `<h2>Saved models from your data</h2><p class="hint">None yet. Tick “learn a model from my own history” when you run a file that includes outcomes.</p>`; return; }
  el.innerHTML = `<h2>Saved models from your data</h2><div class="table-wrap"><table><thead><tr><th>Name</th><th class="num">ROC-AUC</th><th class="num">Trained on</th><th>Saved</th><th></th></tr></thead><tbody>
    ${r.models.map(m => `<tr><td><b>${esc(m.name || m.model_id)}</b><div class="mono muted" style="font-size:12px">${esc(m.model_id)}</div></td><td class="num">${m.metrics ? m.metrics.roc_auc.toFixed(3) : "—"}</td>
      <td class="num">${m.metrics ? fmt(m.metrics.n_train) : "—"}</td><td>${esc((m.saved_at || "").slice(0, 10))}</td>
      <td><label class="btn sm">Score a file<input type="file" accept=".csv,text/csv" class="sr ad-score" data-id="${esc(m.model_id)}"></label> <button class="btn sm ad-del" data-id="${esc(m.model_id)}">Delete</button></td></tr>`).join("")}</tbody></table></div><div id="ad-scored"></div>`;
  $$(".ad-del").forEach(b => b.addEventListener("click", async () => { if (!confirm("Delete this saved model?")) return; try { await api("/api/adapt/models/" + b.dataset.id, { method: "DELETE" }); adaptModelsPanel(); } catch (e) { toast(e.message); } }));
  $$(".ad-score").forEach(inp => inp.addEventListener("change", async () => {
    const f = inp.files[0]; if (!f) return; const fd = new FormData(); fd.append("file", f);
    $("#ad-scored").innerHTML = `<div class="skeleton">Scoring ${esc(f.name)}…</div>`;
    try {
      const s = await api(`/api/adapt/models/${inp.dataset.id}/score`, { method: "POST", body: fd });
      $("#ad-scored").innerHTML = `<div class="callout outcome" style="margin-top:12px"><div class="k">${esc(s.name || "")}</div>${fmt(s.n_rows)} orders scored, ${fmt(s.flagged)} alerts (threshold ${s.threshold}).
        ${s.columns_missing_from_file.length ? `<br><span class="err">Columns missing from this file: ${s.columns_missing_from_file.map(esc).join(", ")}</span>` : ""} <button class="btn sm" id="ad-dl">Download CSV</button></div>`;
      $("#ad-dl").addEventListener("click", () => {
        const cols = ["row", "order_id", "probability", "risk_level", "flagged"];
        const csv = [cols.join(","), ...s.rows.map(x => cols.map(c => x[c]).join(","))].join("\n");
        const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = f.name.replace(/\.csv$/i, "") + "_adapted_scored.csv"; document.body.appendChild(a); a.click(); a.remove();
      });
    } catch (e) { $("#ad-scored").innerHTML = `<div class="callout bad"><div class="k">Could not score</div>${esc(e.message)}</div>`; }
    inp.value = "";
  }));
}

// ------------------------------------------------------------------ ENSEMBLE + TUNING (inside the Model page)
function ensembleHtml(m) {
  if (m.serving === "enriched") {
    const c = m.paired_test_comparison || {};
    return `<section class="panel"><h2>Chronological evaluation · real Olist data</h2>
      <div class="tiles"><div class="tile"><span class="k">Newest-period ROC-AUC</span><span class="v">${Number(m.static_split_auc || 0).toFixed(3)}</span><span class="s">${fmt(m.n_test)} untouched later orders</span></div>
      <div class="tile"><span class="k">PR-AUC</span><span class="v">${Number((m.models.find(x => x.key === m.selected) || {}).pr_auc || 0).toFixed(3)}</span><span class="s">late rate ${(100 * Number(m.test_late_rate || 0)).toFixed(1)}%</span></div>
      <div class="tile"><span class="k">AUC gain vs core ensemble</span><span class="v">${c.paired_auc_difference == null ? "—" : (c.paired_auc_difference >= 0 ? "+" : "") + Number(c.paired_auc_difference).toFixed(3)}</span><span class="s">95% CI ${c.paired_auc_difference_ci95 ? c.paired_auc_difference_ci95.map(x => x.toFixed(3)).join(" to ") : "not available"}</span></div>
      <div class="tile"><span class="k">Monthly backtest mean</span><span class="v">${m.backtest_mean_auc == null ? "—" : m.backtest_mean_auc.toFixed(3)}</span><span class="s">${m.backtest.length} later monthly blocks</span></div></div>
      <p class="note">Calibration warning: ${esc(m.calibration_note || "Risk scores may drift as delivery patterns change.")} The 0.90 ROC-AUC goal was not reached. Olist has no historical weather, carrier scans or hub congestion, so this model does not use them.</p></section>`;
  }
  const e = m.ensemble;
  if (!e) return `<section class="panel"><h2>Ensemble and tuning</h2><p class="hint">Not run in this copy yet. From the project folder: <span class="mono">.venv\\Scripts\\activate</span>, then <span class="mono">python train_ensemble.py</span> (about 20 minutes), then <span class="mono">python backend/add_second_opinion.py</span>, and restart the app.</p></section>`;
  const pb = e.paired_backtest || {};
  const served = e.served;
  const w = Object.entries(e.weights).filter(([, v]) => v > 0);
  const ag = m.agreement;
  return `
    <section class="panel"><h2>Ensemble models and tuning</h2>
      <p class="hint">${e.protocol.map(esc).join(" ")}${e.quick_run ? " <b>This report comes from a quick test run; run the full training for final numbers.</b>" : ""}</p>
      <div class="callout ${served === "blend" ? "outcome" : ""}"><div class="k">Decision · served by default: ${served === "blend" ? "weighted ensemble" : "production XGBoost"}</div>${esc(e.decision)}</div>
      <div class="table-wrap"><table><thead><tr><th>Model</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Top 10% catches</th></tr></thead><tbody>
        ${e.models.map(r => `<tr ${r.served ? 'style="background:var(--accent-soft);font-weight:700"' : ""}><td>${esc(r.label)}${r.served ? " · served" : ""}</td><td class="num">${r.roc_auc.toFixed(3)}</td><td class="num">${r.pr_auc.toFixed(3)}</td>
          <td class="num">${r.precision != null ? pct(r.precision) : "—"}</td><td class="num">${r.recall != null ? pct(r.recall) : "—"}</td><td class="num">${r.f1 != null ? r.f1.toFixed(3) : "—"}</td><td class="num">${r.recall_at_top10pct != null ? pct(r.recall_at_top10pct) : "—"}</td></tr>`).join("")}
        <tr class="muted"><td>${esc(e.stack.label)}</td><td class="num">${e.stack.roc_auc.toFixed(3)}</td><td class="num">${e.stack.pr_auc.toFixed(3)}</td><td colspan="4"></td></tr></tbody></table></div>
      <p class="note">Each model's alert threshold is the F1-optimal cut-off on training-set predictions, never on the test orders. “Top 10% catches” is the share of all late orders that sit in the 10% of orders ranked riskiest.</p></section>
    <div class="grid g2">
      <section class="panel"><h2>Retrain every month: single model versus ensemble</h2>
        <p class="hint">Mean ROC-AUC: ${Object.entries(e.backtest_mean_auc).map(([k, v]) => `${esc(k.replace("production_xgb", "production XGBoost").replace("xgb_tuned", "re-tuned XGBoost").replace("blend", "ensemble"))} ${v}`).join(" · ")}${pb.months ? `. The ensemble is ahead of the production XGBoost in ${pb.months_blend_better} of ${pb.months} months.` : ""}</p>
        <div class="chart"><canvas id="m-ens-bt"></canvas></div></section>
      <section class="panel"><h2>What tuning changed</h2>
        <p class="hint">${e.tuning.xgb.trials} XGBoost settings tried (the production settings are trial 0). Cross-validated ROC-AUC on the training orders: current settings ${e.tuning.xgb.baseline_cv_auc.toFixed(4)}, best found ${e.tuning.xgb.best_cv_auc.toFixed(4)}.</p>
        <div class="chart"><canvas id="m-tune"></canvas></div>
        <p class="note">Most settings land within a few thousandths of each other. When many different settings give the same score, the data, not the tuning, is the limit, which is why the ensemble and the monthly check matter more than another decimal.</p></section>
    </div>
    ${m.calibration ? `<section class="panel"><h2>Is the score a real chance?</h2>
      <p class="hint">No: the models are trained to rank late orders above on-time ones (class-balanced), so a score of 0.7 does not mean 70%. A calibration step turns the ensemble's score into the share of such orders that were really late. Expected calibration error on unseen orders: <b>${(m.calibration.test.ece_calibrated * 100).toFixed(1)} percentage points</b>, against ${(m.calibration.test.ece_raw_score * 100).toFixed(0)} if the raw score were read as a probability. Brier score ${m.calibration.test.brier_calibrated} (guessing the average gives ${m.calibration.test.base_rate_brier}).</p>
      <div class="chart"><canvas id="m-cal"></canvas></div></section>` : ""}
    <div class="grid g2">
      <section class="panel"><h2>Ensemble members</h2>
        <div class="chart" style="height:${40 + w.length * 36}px"><canvas id="m-weights"></canvas></div>
        <p class="note">Weights fitted on out-of-fold training predictions. Reasons shown on each order are exact TreeSHAP values from the XGBoost member.</p></section>
      <section class="panel"><h2>Do the two models agree on the queue?</h2>
        ${ag ? `<div class="tiles" style="grid-template-columns:repeat(2,1fr)"><div class="tile"><span class="k">Both alert</span><span class="v">${fmt(ag.both)}</span><span class="s">${ag.agree_share}% of all alerts</span></div>
          <div class="tile"><span class="k">Only one alerts</span><span class="v">${fmt(ag.n_disagree)}</span><span class="s">${esc(ag.serving_model || "XGBoost")} only ${fmt(ag.serving_only ?? ag.only_xgboost)} · ${esc(ag.other_model || "ensemble")} only ${fmt(ag.other_only ?? ag.only_ensemble)}</span></div></div>
          <a class="btn" href="#/queue?disagree=true">Open those orders in the queue</a>` : `<p class="hint">Run <span class="mono">python backend/add_second_opinion.py</span> to score the queue with both models.</p>`}
        <p class="note">Models that disagree are worth a human look: it is where the second opinion adds information.</p></section>
    </div>`;
}

function ensembleCharts(m) {
  const e = m.ensemble; if (!e) return;
  const keys = Object.keys(e.backtest || {});
  if (keys.length && $("#m-ens-bt")) {
    const months = e.backtest[keys[0]].map(r => r.month);
    const nm = { production_xgb: "Production XGBoost", xgb_tuned: "Re-tuned XGBoost", blend: "Ensemble" };
    const cols = { production_xgb: css("--faint"), xgb_tuned: css("--chart-2") || css("--med"), blend: css("--chart-1") };
    chart($("#m-ens-bt"), { type: "line", data: { labels: months, datasets: keys.map(k => ({ label: nm[k] || k, data: months.map(mo => (e.backtest[k].find(r => r.month === mo) || {}).auc ?? null), borderColor: cols[k], backgroundColor: cols[k], pointRadius: 2, tension: .2, borderWidth: k === "blend" ? 3 : 1.5 })) },
      options: { scales: { y: { min: 0.4, max: 0.9 }, x: { grid: { display: false } } }, plugins: { legend: { display: true, position: "bottom" } } } });
  }
  if ($("#m-tune")) {
    const all = e.tuning.xgb.all_cv_auc;
    chart($("#m-tune"), { type: "bar", data: { labels: all.map((_, i) => i + 1), datasets: [{ data: all, backgroundColor: all.map(v => v === e.tuning.xgb.baseline_cv_auc ? css("--high") : css("--chart-1")), borderRadius: 2 }] },
      options: { scales: { y: { min: Math.floor(Math.min(...all) * 50) / 50, title: { display: true, text: "CV ROC-AUC" } }, x: { title: { display: true, text: "trials, best first" }, grid: { display: false } } },
        plugins: { tooltip: { callbacks: { label: c => `CV ROC-AUC ${c.raw}${c.raw === e.tuning.xgb.baseline_cv_auc ? " (current settings)" : ""}` } } } } });
  }
  if ($("#m-cal") && m.calibration) {
    const rel = m.calibration.reliability_calibrated, raw = m.calibration.reliability_raw;
    chart($("#m-cal"), { type: "line", data: { datasets: [
      { label: "Perfect", data: [{ x: 0, y: 0 }, { x: 0.6, y: 0.6 }], borderColor: css("--faint"), borderDash: [5, 4], pointRadius: 0, borderWidth: 1 },
      { label: "Calibrated chance", data: rel.map(r => ({ x: r.predicted, y: r.observed })), borderColor: css("--chart-1"), backgroundColor: css("--chart-1"), pointRadius: 4, borderWidth: 3 },
      { label: "Raw score", data: raw.map(r => ({ x: r.predicted, y: r.observed })), borderColor: css("--high"), backgroundColor: css("--high"), pointRadius: 3, borderWidth: 1.5 }] },
      options: { parsing: false, scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "predicted" } }, y: { min: 0, max: 0.6, title: { display: true, text: "share really late" } } }, plugins: { legend: { display: true, position: "bottom" } } } });
  }
  if ($("#m-weights")) {
    const w = Object.entries(e.weights).filter(([, v]) => v > 0);
    hbar($("#m-weights"), w.map(([k]) => e.member_labels[k] || k), w.map(([, v]) => +(v * 100).toFixed(0)), css("--chart-1"), { unit: "%", thick: 18 });
  }
}


// ------------------------------------------------------------------ nav: collapsible groups + phone tab bar
function initNavGroups() {
  const nav = document.getElementById("nav"); if (!nav) return;
  const start = new Set(["Monitor", "Act"]);
  let cur = null;
  [...nav.children].forEach(el => {
    if (el.classList.contains("group")) {
      const name = el.textContent.trim();
      const b = document.createElement("button");
      b.type = "button"; b.className = "group"; b.dataset.group = name;
      b.innerHTML = `<span>${esc(name)}</span><svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
      b.setAttribute("aria-expanded", start.has(name) ? "true" : "false");
      b.addEventListener("click", () => setGroup(b, b.getAttribute("aria-expanded") !== "true"));
      el.replaceWith(b); cur = b;
    } else if (cur) { el.dataset.group = cur.dataset.group; }
  });
  nav.querySelectorAll("a[data-group]").forEach(a => {
    const g = nav.querySelector(`button.group[data-group="${a.dataset.group}"]`);
    a.classList.toggle("nav-hid", g.getAttribute("aria-expanded") !== "true");
  });
  const cp = (location.hash.replace(/^#\/?/, "").split(/[?\/]/)[0]) || "overview";
  revealActiveGroup(cp);
}
function setGroup(btn, open) {
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  document.querySelectorAll(`#nav a[data-group="${btn.dataset.group}"]`).forEach(a => a.classList.toggle("nav-hid", !open));
}
function revealActiveGroup(page) {
  const a = document.querySelector(`#nav a[data-page="${page}"]`); if (!a || !a.dataset.group) return;
  const b = document.querySelector(`#nav button.group[data-group="${a.dataset.group}"]`);
  if (b && b.getAttribute("aria-expanded") !== "true") setGroup(b, true);
}
function initTabbar() {
  const more = document.getElementById("tab-more"); if (!more) return;
  const close = () => { document.body.classList.remove("nav-open"); more.setAttribute("aria-expanded", "false"); };
  more.addEventListener("click", () => { const o = document.body.classList.toggle("nav-open"); more.setAttribute("aria-expanded", String(o)); });
  document.getElementById("nav").addEventListener("click", e => { if (e.target.closest("a")) close(); });
  document.getElementById("scrim-nav").addEventListener("click", close);
  addEventListener("keydown", e => { if (e.key === "Escape") close(); });
  addEventListener("hashchange", close);
  const mirror = () => { [["nav-flagged", "tab-flagged"], ["nav-board", "tab-board"]].forEach(([a, b]) => { const s = document.getElementById(a), d = document.getElementById(b); if (s && d) d.textContent = s.textContent; }); };
  const mo = new MutationObserver(mirror);
  ["nav-flagged", "nav-board"].forEach(id => { const el = document.getElementById(id); if (el) mo.observe(el, { childList: true, characterData: true, subtree: true }); });
  mirror();
}

// ------------------------------------------------------------------ boot
function applyThemeLabel() {
  const t = currentTheme();
  $("#theme-label").textContent = t.mode === "dark" ? "Light version" : "Dark version";
}
$("#theme-btn").addEventListener("click", () => setTheme(currentTheme().pair));
$("#search").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#search-input").value.trim();
  if (q) go("queue", new URLSearchParams({ q }));
});
window.addEventListener("hashchange", route);

(async function boot() {
  applyThemeLabel();
  renderRailSwatches();
  initNavGroups();
  initTabbar();
  try {
    S.meta = await api("/api/meta");
    const h = await api("/api/health");
    $("#status").textContent = `${(S.meta.queue_view || {}).label || "Model"} · ${fmt(h.orders_in_queue)} orders`;
    const s = await api("/api/summary");
    $("#nav-flagged").textContent = fmt(s.n_flagged);
    refreshCounts();
  } catch (e) {
    const offline = /Failed to fetch|NetworkError|Load failed/i.test(e.message);
    $("#status").textContent = offline ? "Server not reachable" : "Server problem";
    view.innerHTML = offline
      ? `<div class="panel"><h2>Can't reach the OnTime server</h2><p class="hint">Start it with <span class="mono">start_ontime.bat</span> (or <span class="mono">python run.py</span>) and keep that window open, then reload this page.</p></div>`
      : `<div class="panel"><h2>OnTime couldn't start</h2><p class="err">${esc(e.message)}</p><p class="hint">The console window shows more detail. After fixing it, restart start_ontime.bat and reload this page.</p></div>`;
    return;
  }
  route();
})();
})();
