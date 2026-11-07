fetch("data/thresholds.json")
  .then(r => r.json())
  .then(init);

function init(d) {
  const sel = document.getElementById("model");
  const names = Object.keys(d.validation);
  names.forEach(n => sel.add(new Option(n, n)));
  const slider = document.getElementById("t");

  function draw() {
    const name = sel.value;
    const t = parseFloat(slider.value);
    const row = d.validation[name].reduce((a, b) =>
      Math.abs(b.threshold - t) < Math.abs(a.threshold - t) ? b : a);
    document.getElementById("tval").textContent = t.toFixed(2);
    document.getElementById("tp").textContent = row.TP;
    document.getElementById("fp").textContent = row.FP;
    document.getElementById("fn").textContent = row.FN;
    document.getElementById("tn").textContent = row.TN;
    document.getElementById("prec").textContent = row.precision.toFixed(3);
    document.getElementById("rec").textContent = row.recall.toFixed(3);
    const f = d.frozen_thresholds[name];
    document.getElementById("frozen").textContent =
      "Frozen threshold for this model: " + f.toFixed(3) +
      (Math.abs(f - t) < 0.01 ? "  (you are at it)" : "");
    const p = d.persistence;
    document.getElementById("pers").textContent =
      "Persistence (tomorrow = today) for comparison: precision " +
      p.precision.toFixed(3) + ", recall " + p.recall.toFixed(3);
  }
  sel.onchange = draw;
  slider.oninput = draw;
  slider.value = d.frozen_thresholds[names[0]];
  draw();
}