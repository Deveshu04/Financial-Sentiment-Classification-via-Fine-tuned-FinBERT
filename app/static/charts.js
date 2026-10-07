const SVG_NS = "http://www.w3.org/2000/svg";

function svgNode(name, attrs, parent) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
  if (parent) parent.append(node);
  return node;
}

function svgText(parent, x, y, content, attrs) {
  const node = svgNode("text", {x, y, ...(attrs || {})}, parent);
  node.textContent = content;
  return node;
}

function linear(d0, d1, r0, r1) {
  return (value) => r0 + ((value - d0) / (d1 - d0 || 1)) * (r1 - r0);
}

function niceTicks(lo, hi, count) {
  const raw = (hi - lo) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  const step = (unit < 1.5 ? 1 : unit < 3 ? 2 : unit < 7 ? 5 : 10) * power;
  const ticks = [];
  for (let value = Math.ceil(lo / step) * step; value <= hi + step * 1e-6; value += step) ticks.push(Number(value.toFixed(10)));
  return ticks;
}

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function canvas(container, height) {
  container.replaceChildren();
  const width = Math.max(container.clientWidth, 280);
  const svg = svgNode("svg", {viewBox: `0 0 ${width} ${height}`, width, height, class: "plot"}, container);
  const tooltip = document.createElement("div");
  tooltip.className = "tooltip";
  tooltip.hidden = true;
  container.append(tooltip);
  return {svg, width, tooltip};
}

function showTooltip(tooltip, container, x, y, lines) {
  tooltip.replaceChildren();
  lines.forEach((line, i) => {
    if (i) tooltip.append(document.createElement("br"));
    const part = document.createElement(i ? "span" : "b");
    part.textContent = line;
    tooltip.append(part);
  });
  tooltip.hidden = false;
  const left = Math.min(Math.max(x + 12, 0), container.clientWidth - tooltip.offsetWidth);
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${Math.max(y - tooltip.offsetHeight - 10, 0)}px`;
}

function dotPlot(container, rows, options) {
  const {format, references = [], tickFormat = format} = options;
  const step = 30;
  const groups = rows.filter((row, i) => row.group && (i === 0 || rows[i - 1].group !== row.group)).length;
  const narrow = container.clientWidth < 560;
  const margin = {top: 26, right: 64, bottom: 30, left: narrow ? Math.min(150, container.clientWidth * 0.42) : Math.min(260, container.clientWidth * 0.36)};
  const height = margin.top + margin.bottom + (rows.length + groups) * step;
  const {svg, width, tooltip} = canvas(container, height);
  const values = rows.flatMap((row) => [row.low, row.high]).concat(references.map((ref) => ref.value));
  const lo = Math.min(...values) - 0.02;
  const hi = Math.max(...values) + 0.02;
  const x = linear(lo, hi, margin.left, width - margin.right);
  niceTicks(lo, hi, narrow ? 3 : 5).forEach((tick) => {
    svgNode("line", {x1: x(tick), x2: x(tick), y1: margin.top, y2: height - margin.bottom, class: "grid"}, svg);
    svgText(svg, x(tick), height - margin.bottom + 18, tickFormat(tick), {class: "tick", "text-anchor": "middle"});
  });
  references.forEach((ref, i) => {
    svgNode("line", {x1: x(ref.value), x2: x(ref.value), y1: margin.top - 8, y2: height - margin.bottom, class: `reference ${ref.style || ""}`}, svg);
    svgText(svg, x(ref.value), margin.top - 12, ref.label, {class: "reference-label", "text-anchor": i % 2 ? "start" : "end", dx: i % 2 ? 4 : -4});
  });
  let y = margin.top + step / 2;
  rows.forEach((row, i) => {
    if (row.group && (i === 0 || rows[i - 1].group !== row.group)) {
      svgText(svg, 0, y + 5, row.group, {class: "group-label"});
      y += step;
    }
    const label = svgText(svg, margin.left - 12, y + 5, row.label, {class: "row-label", "text-anchor": "end"});
    const room = margin.left - 16;
    if (label.getComputedTextLength() > room) {
      let text = row.label;
      while (text.length > 4 && label.getComputedTextLength() > room) {
        text = text.slice(0, -1);
        label.textContent = `${text.trimEnd()}...`;
      }
    }
    svgNode("line", {x1: x(row.low), x2: x(row.high), y1: y, y2: y, class: "whisker", stroke: row.color}, svg);
    const dot = svgNode("circle", {cx: x(row.value), cy: y, r: 5, fill: row.color, class: "dot"}, svg);
    svgText(svg, x(row.high) + 8, y + 5, format(row.value), {class: "value-label"});
    const hit = svgNode("rect", {x: margin.left, y: y - step / 2, width: width - margin.left - margin.right, height: step, fill: "transparent"}, svg);
    const describe = [row.label, `${format(row.value)}, 95% interval ${format(row.low)} to ${format(row.high)}`];
    hit.addEventListener("pointerenter", () => showTooltip(tooltip, container, Number(dot.getAttribute("cx")), Number(dot.getAttribute("cy")), describe));
    hit.addEventListener("pointerleave", () => { tooltip.hidden = true; });
    y += step;
  });
}

function lineChart(container, dates, series, options) {
  const {format, references = [], height = 240} = options;
  const margin = {top: 34, right: 18, bottom: 28, left: 52};
  const {svg, width, tooltip} = canvas(container, height);
  const times = dates.map((d) => Date.parse(`${d}T00:00:00Z`));
  const values = series.flatMap((s) => s.values.filter((v) => v !== null)).concat(references.map((ref) => ref.value));
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const pad = (hi - lo) * 0.08 || 0.01;
  lo -= pad;
  hi += pad;
  const x = linear(times[0], times[times.length - 1], margin.left, width - margin.right);
  const y = linear(lo, hi, height - margin.bottom, margin.top);
  niceTicks(lo, hi, 4).forEach((tick) => {
    svgNode("line", {x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick), class: "grid"}, svg);
    svgText(svg, margin.left - 8, y(tick) + 4, format(tick), {class: "tick", "text-anchor": "end"});
  });
  const labels = Math.max(2, Math.floor((width - margin.left - margin.right) / 110));
  const dateLabel = (time) => new Date(time).toLocaleDateString("en-GB", {month: "short", year: "numeric", timeZone: "UTC"});
  for (let i = 0; i < labels; i += 1) {
    const index = Math.round((i / (labels - 1)) * (dates.length - 1));
    svgText(svg, x(times[index]), height - margin.bottom + 18, dateLabel(times[index]), {class: "tick", "text-anchor": i === 0 ? "start" : i === labels - 1 ? "end" : "middle"});
  }
  references.forEach((ref) => {
    svgNode("line", {x1: margin.left, x2: width - margin.right, y1: y(ref.value), y2: y(ref.value), class: `reference ${ref.style || ""}`}, svg);
    if (ref.label) svgText(svg, width - margin.right, y(ref.value) - 5, ref.label, {class: "reference-label", "text-anchor": "end"});
  });
  let legendX = margin.left;
  series.forEach((s) => {
    let d = "";
    let pen = false;
    s.values.forEach((value, i) => {
      if (value === null) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${x(times[i]).toFixed(1)},${y(value).toFixed(1)}`;
      pen = true;
    });
    svgNode("path", {d, class: "series", stroke: s.color}, svg);
    if (series.length > 1) {
      svgNode("rect", {x: legendX, y: 9, width: 14, height: 3, rx: 1.5, fill: s.color}, svg);
      const label = svgText(svg, legendX + 20, 15, s.label, {class: "legend-label"});
      legendX += 36 + label.getComputedTextLength();
    } else {
      svgText(svg, margin.left, 15, s.label, {class: "legend-label"});
    }
  });
  const cross = svgNode("line", {y1: margin.top, y2: height - margin.bottom, class: "crosshair", visibility: "hidden"}, svg);
  const dots = series.map((s) => svgNode("circle", {r: 4, fill: s.color, class: "dot", visibility: "hidden"}, svg));
  const area = svgNode("rect", {x: margin.left, y: margin.top, width: width - margin.left - margin.right, height: height - margin.top - margin.bottom, fill: "transparent"}, svg);
  area.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * width;
    const time = times[0] + ((px - margin.left) / (width - margin.left - margin.right)) * (times[times.length - 1] - times[0]);
    let index = 0;
    times.forEach((t, i) => { if (Math.abs(t - time) < Math.abs(times[index] - time)) index = i; });
    const cx = x(times[index]);
    cross.setAttribute("x1", cx);
    cross.setAttribute("x2", cx);
    cross.setAttribute("visibility", "visible");
    const lines = [new Date(times[index]).toLocaleDateString("en-GB", {day: "numeric", month: "short", year: "numeric", timeZone: "UTC"})];
    series.forEach((s, i) => {
      const value = s.values[index];
      dots[i].setAttribute("visibility", value === null ? "hidden" : "visible");
      if (value !== null) {
        dots[i].setAttribute("cx", cx);
        dots[i].setAttribute("cy", y(value));
      }
      lines.push(`${s.label}: ${value === null ? "no value" : format(value)}`);
    });
    showTooltip(tooltip, container, cx, margin.top + 10, lines);
  });
  area.addEventListener("pointerleave", () => {
    tooltip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dots.forEach((dot) => dot.setAttribute("visibility", "hidden"));
  });
}

function onRedraw(draw) {
  draw();
  if (document.fonts) document.fonts.ready.then(draw);
  let pending;
  window.addEventListener("resize", () => {
    clearTimeout(pending);
    pending = setTimeout(draw, 150);
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
}
