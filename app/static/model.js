const modelData = JSON.parse(document.getElementById("page-data").textContent);
const MODEL_KEYS = ["finbert", "baseline", "prosus"];
const MODEL_TOKENS = {finbert: "--series-1", baseline: "--series-2", prosus: "--series-3"};
const asPercent = (value) => `${(value * 100).toFixed(1)}%`;
const SHORT = {finbert: "Fine-tuned FinBERT", baseline: "TF-IDF + LR", prosus: "ProsusAI reference"};

onRedraw(() => {
  const groups = [{name: `Full test split (n = ${modelData.test_n})`, models: modelData.full}].concat(
    modelData.slices.map((item) => ({name: `${item.name} (n = ${item.n})`, models: Object.fromEntries(MODEL_KEYS.map((key) => [key, item.models[key].accuracy]))}))
  );
  const rows = groups.flatMap((group) => MODEL_KEYS.map((key) => ({group: group.name, label: SHORT[key], value: group.models[key].value, low: group.models[key].low, high: group.models[key].high, color: token(MODEL_TOKENS[key])})));
  dotPlot(document.getElementById("slices"), rows, {format: asPercent, tickFormat: (v) => `${Math.round(v * 100)}%`, references: [{value: modelData.target, label: `${asPercent(modelData.target)} target`, style: "target"}]});
});
