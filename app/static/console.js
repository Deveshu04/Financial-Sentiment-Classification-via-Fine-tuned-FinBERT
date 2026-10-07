const input = document.getElementById("text-input");
const button = document.getElementById("score");
const statusLine = document.getElementById("status");
const results = document.getElementById("results");
const LABELS = ["negative", "neutral", "positive"];

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function resultItem(text, prediction) {
  const item = document.createElement("li");
  item.className = "result";
  const sentence = document.createElement("p");
  sentence.textContent = text;
  const verdict = document.createElement("p");
  verdict.className = "verdict";
  const label = document.createElement("strong");
  label.textContent = prediction.label;
  verdict.append("Read as ", label, ` (${percent(prediction.probabilities[prediction.label])})`);
  const bar = document.createElement("div");
  bar.className = "tone";
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", LABELS.map((name) => `${name} ${percent(prediction.probabilities[name])}`).join(", "));
  const values = document.createElement("ul");
  values.className = "tone-values";
  LABELS.forEach((name) => {
    const segment = document.createElement("span");
    segment.className = `segment ${name}`;
    segment.style.flexGrow = prediction.probabilities[name];
    bar.append(segment);
    const value = document.createElement("li");
    value.className = name;
    value.textContent = `${name} ${percent(prediction.probabilities[name])}`;
    values.append(value);
  });
  item.append(sentence, verdict, bar, values);
  return item;
}

async function score() {
  const texts = input.value.split("\n").map((line) => line.trim()).filter(Boolean);
  if (!texts.length) {
    statusLine.textContent = "Enter at least one sentence to score.";
    return;
  }
  button.disabled = true;
  statusLine.textContent = "Scoring...";
  try {
    const response = await fetch("/api/sentiment", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({texts})});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || response.statusText);
    results.replaceChildren(...body.predictions.map((prediction, i) => resultItem(texts[i], prediction)));
    statusLine.textContent = `Scored ${body.predictions.length} ${body.predictions.length === 1 ? "sentence" : "sentences"} in ${body.latency_ms.toFixed(0)} ms of server time.`;
  } catch (error) {
    statusLine.textContent = `Could not score the text: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

document.querySelectorAll(".example").forEach((example) => {
  example.addEventListener("click", () => {
    input.value = example.dataset.example;
    score();
  });
});
button.addEventListener("click", score);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) score();
});
