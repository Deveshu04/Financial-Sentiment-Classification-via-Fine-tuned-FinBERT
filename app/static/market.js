const marketData = JSON.parse(document.getElementById("page-data").textContent);
const ORDER = ["sentiment", "sentiment_classifier", "sentiment_returns", "returns_only", "previous_direction", "always_up"];
const pct = (value) => `${(value * 100).toFixed(1)}%`;
const SHORT = {sentiment: "Headline model", sentiment_classifier: "Classifier", sentiment_returns: "With returns", returns_only: "Returns only", previous_direction: "Previous session", always_up: "Always up"};

onRedraw(() => {
  const target = marketData.benchmark;
  const headline = token("--series-1");
  const muted = token("--muted");
  dotPlot(document.getElementById("contrasts"), ORDER.map((key) => ({label: SHORT[key], value: marketData.models[key].accuracy, low: marketData.models[key].low, high: marketData.models[key].high, color: key === "sentiment" ? headline : muted})), {format: pct, tickFormat: (v) => `${Math.round(v * 100)}%`, references: [{value: 0.5, label: "50%"}, {value: target, label: `${pct(target)} target`, style: "target"}]});
  lineChart(document.getElementById("rolling"), marketData.rolling.dates, [
    {label: `Headline model, ${marketData.rolling.window}-session accuracy`, values: marketData.rolling.accuracy, color: headline},
    {label: "Share of up sessions", values: marketData.rolling.always_up, color: muted},
  ], {format: pct, references: [{value: 0.5, label: "50%"}, {value: target, label: `${pct(target)} target`, style: "target"}]});
  const tone = document.getElementById("tone");
  tone.replaceChildren();
  const upper = document.createElement("div");
  const lower = document.createElement("div");
  upper.className = "chart";
  lower.className = "chart";
  tone.append(upper, lower);
  lineChart(upper, marketData.tone.dates, [{label: "Headline net tone, 5-session mean", values: marketData.tone.net_tone_r5, color: headline}], {format: (v) => v.toFixed(2), references: [{value: 0, label: ""}], height: 200});
  lineChart(lower, marketData.tone.dates, [{label: "S&P 500 open to close return %, 5-session mean", values: marketData.tone.target_r5, color: token("--ink-2")}], {format: (v) => v.toFixed(1), references: [{value: 0, label: ""}], height: 200});
});
