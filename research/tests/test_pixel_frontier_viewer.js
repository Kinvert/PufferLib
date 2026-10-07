/* Synthetic display fixtures only; no policy, reference model or GPU. */
"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const F = require("../pixel_frontier.js");

function fixture() {
    const condition = {environment: "pongcnn", training_appearance: "mixed-s42-c7", evaluation_representation: 6,
        status: "complete_descriptive", metric: "mean final point fraction (censoring bounds)", observed: 16, expected: 16, failures: 0};
    const data = {status: "complete_descriptive", conditions: [condition], planned_cells: [], points: [], means: [],
        missing: [], failures: [], training_job_costs: [], accounted_training_process_seconds: 20,
        training_cost_accounting_complete: true, qualification: "SYNTHETIC, not model validation", timing: "synthetic clock"};
    for (const [m, model] of F.models.entries()) {
        for (const seed of [11, 12]) for (const steps of [100, 200]) {
            const row = {environment: condition.environment, training_appearance: condition.training_appearance,
                evaluation_representation: 6, model, seed, steps, job_index: m * 2 + seed - 11,
                binding_index: m * 2 + seed - 11};
            data.planned_cells.push(row);
            data.points.push({...row, seconds: steps / 100 * (m + 1), score_lower: steps === 100 ? .4 : .2,
                score_upper: steps === 100 ? .9 : .7, parameters: 4, episodes: 3});
        }
        for (const steps of [100, 200]) data.means.push({environment: condition.environment,
            training_appearance: condition.training_appearance, evaluation_representation: 6, model, steps,
            seconds: steps / 100 * (m + 1), score_lower: steps === 100 ? .4 : .2,
            score_upper: steps === 100 ? .9 : .7, seeds: 2,
            frontier_seconds_lower: steps === 100 && m === 0, frontier_seconds_upper: steps === 100 && m === 0});
    }
    return data;
}

test("all four models and both seeds remain allocated; hiding does not change coverage or means", () => {
    const data = fixture(), before = JSON.stringify(data), selected = F.select(data, data.conditions[0], "mean", ["nature_cnn"]);
    assert.equal(selected.points.length, 2); assert.equal(selected.means.length, 8);
    assert.equal(selected.observations.length, 16); assert.deepEqual(selected.seeds, [11, 12]);
    assert.equal(JSON.stringify(data), before); assert.ok(!selected.points[0].frontier_seconds_lower);
});
test("seed selection never becomes a best-seed envelope", () => {
    const data = fixture(), selected = F.select(data, data.conditions[0], "12");
    assert.equal(selected.points.length, 8); assert.ok(selected.points.every(row => row.seed === 12));
});
test("arbitrary declared candidates and hyphenated references remain selectable", () => {
    const data = fixture(), names = ["happy-cat-1", "nature-cnn", "impala-cnn", "impoola-cnn"];
    for (const field of ["planned_cells", "points", "means"]) {
        for (const row of data[field]) row.model = names[F.models.indexOf(row.model)];
    }
    data.model_catalog = names;
    const catalog = F.catalog(data);
    assert.deepEqual(catalog.models, names); assert.equal(catalog.labels["impala-cnn"], "IMPALA");
    assert.equal(F.select(data, data.conditions[0], "mean").points.length, 8);
    const color = catalog.colors["happy-cat-1"];
    assert.match(color, /^hsl\([0-9]+ 70% 38%\)$/);
    assert.equal(F.catalog({...data, model_catalog: [...names].reverse()}).colors["happy-cat-1"], color);
    const elements = application(data);
    assert.equal(elements.legend.children.length, 4); assert.equal(elements.points.children.length, 8);
    assert.equal(elements.coverage.children[0].children[0].textContent, "happy-cat-1");
    assert.equal(elements.coverage.children[2].children[0].textContent, "IMPALA");
});
test("declared missing candidates cannot be omitted or replaced by unknown observations", () => {
    const data = fixture();
    for (const model_catalog of ["invalid", ["flex_quality"], [...F.models, "ghost"], [...F.models, "flex_quality"]]) {
        assert.throws(() => F.catalog({...data, model_catalog}));
    }
    assert.throws(() => F.catalog({...data, points: [...data.points, {...data.points[0], model: "unallocated"}]}));
    data.points = []; data.means = [];
    assert.deepEqual(F.catalog(data).models, F.models);
});
test("a fifth unobserved candidate stays in the legend and coverage when models are hidden", () => {
    const data = fixture(); data.model_catalog = [...F.models, "mystic-tree-2"];
    for (const seed of [11, 12]) for (const steps of [100, 200]) {
        data.planned_cells.push({...data.planned_cells[0], model: "mystic-tree-2", seed, steps,
            job_index: 8+seed-11, binding_index: 8+seed-11});
    }
    data.conditions[0].status = "incomplete"; data.conditions[0].expected = 20;
    data.missing = data.planned_cells.filter(row => row.model === "mystic-tree-2")
        .map(row => ({binding_index: row.binding_index, steps: row.steps}));
    const elements = application(data);
    assert.equal(elements.legend.children.length, 5); assert.equal(elements.coverage.children.length, 5);
    assert.equal(elements.coverage.children[4].children[0].textContent, "mystic-tree-2");
    assert.equal(elements.coverage.children[4].children[2].textContent, 0);
    assert.equal(elements.missing.children.length, 4);
    const first = elements.legend.children[0].children[0]; first.checked = false; first.events.change();
    assert.equal(elements.coverage.children.length, 5); assert.equal(elements.points.children.length, 6);
    assert.equal(elements.missing.children.length, 4);
});
test("model names are text, including prototype-like and markup names", () => {
    const data = fixture(), names = ["__proto__", "constructor", "</script><img src=x>", "plain-model"];
    for (const field of ["planned_cells", "points", "means"]) {
        for (const row of data[field]) row.model = names[F.models.indexOf(row.model)];
    }
    data.model_catalog = names;
    const catalog = F.catalog(data); assert.equal(catalog.labels["__proto__"], "__proto__");
    const elements = application(data);
    assert.equal(elements.coverage.children[2].children[0].textContent, names[2]);
    assert.ok(elements.legend.children.every(label => label.children.length === 2));
});
test("drawing/training treatment stay separate and shared job costs are not duplicated", () => {
    const data = fixture(), original = data.conditions[0];
    data.planned_cells.push({...data.planned_cells[0], evaluation_representation: 0});
    data.points.push({...data.points[0], evaluation_representation: 0, score_lower: 1});
    data.points.push({...data.points[0], training_appearance: "fixed-r0"});
    data.training_job_costs.push({job_index: 0, process_seconds: 20});
    const selected = F.select(data, original, "mean");
    assert.equal(selected.planned.length, 16); assert.equal(selected.observations.length, 16);
    assert.equal(selected.costs.length, 1); assert.equal(selected.costs[0].process_seconds, 20);
});
test("missing cells and training/evaluation failures preserve their condition identities", () => {
    const data = fixture(), removed = data.points.pop();
    data.failures.push({binding_index: removed.binding_index, steps: removed.steps, error: "timeout"},
        {job_index: 0, error: "failed training"}, {job_index: 99, error: "unrelated"});
    const selected = F.select(data, data.conditions[0], "mean");
    assert.equal(selected.missing.length, 1); assert.equal(selected.missing[0].steps, removed.steps);
    assert.equal(selected.failures.length, 2);
});
test("chronological declines stay connected but missing intermediate checkpoints break curves", () => {
    const rows = [{steps: 300, score_lower: .2}, {steps: 100, score_lower: .9}];
    assert.deepEqual(F.segments(rows, [100, 200, 300]), [[rows[1]], [rows[0]]]);
    assert.deepEqual(F.segments(rows, [100, 300]), [[rows[1], rows[0]]]);
});
test("front marks require stored true flags, complete condition, means and time axis", () => {
    const data = fixture(), row = data.means[0], condition = data.conditions[0];
    assert.ok(F.marked(row, condition, "mean", "seconds", "lower"));
    for (const flag of [null, false, 1, "true"]) assert.equal(F.marked({...row, frontier_seconds_lower: flag}, condition, "mean", "seconds", "lower"), false);
    assert.equal(F.marked(row, {...condition, status: "incomplete"}, "mean", "seconds", "lower"), false);
    assert.equal(F.marked(row, condition, "11", "seconds", "lower"), false);
    assert.equal(F.marked(row, condition, "mean", "steps", "lower"), false);
});
test("scales include upper censoring bounds and full costs without a midpoint or time cutoff", () => {
    assert.deepEqual(F.bounds([{seconds: 10000, score_lower: .1, score_upper: .8}], "seconds", true), {xmax: 10500, ymax: 1});
    assert.equal(F.bounds([{steps: 500, score_upper: 5000}], "steps", false).ymax, 5250);
    assert.deepEqual(F.bounds([], "seconds", true), {xmax: 1.05, ymax: 1});
});

// Minimal DOM exercises actual embedded application/event handlers, not a browser
// layout engine. All text/series fixtures are synthetic; no network/CDN needed.
class Element {
    constructor(tag) {this.tag = tag; this.children = []; this.events = {}; this.attrs = {}; this._value = ""; this.textContent = ""; this.style = {};}
    set value(value) {this._value = String(value);}
    get value() {return this._value;}
    appendChild(node) {this.children.push(node); if (this.tag === "select" && this.children.length === 1) this.value = node.value; return node;}
    replaceChildren() {this.children = []; if (this.tag === "select") this.value = "";}
    setAttribute(key, value) {this.attrs[key] = value;}
    addEventListener(name, callback) {this.events[name] = callback;}
}
function application(data) {
    const html = fs.readFileSync(path.join(__dirname, "../pixel_frontier.html"), "utf8");
    const ids = [...html.matchAll(/id="([a-z]+)"/g)].map(match => match[1]);
    const elements = Object.fromEntries(ids.map(id => [id, new Element(["condition", "view", "axis"].includes(id) ? "select" : id)]));
    elements.data.textContent = JSON.stringify(data); elements.axis.value = "seconds"; elements.raw.checked = false;
    const document = {getElementById: id => elements[id], createElement: tag => new Element(tag),
        createElementNS: (_, tag) => new Element(tag), createTextNode: text => Object.assign(new Element("text"), {textContent: text})};
    const scripts = [...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)];
    const context = vm.createContext({document});
    vm.runInContext(fs.readFileSync(path.join(__dirname, "../pixel_frontier.js"), "utf8"), context);
    vm.runInContext(scripts.at(-1)[1], context);
    return elements;
}
test("actual application renders complete censored curves, every model and selectable seeds", () => {
    const elements = application(fixture());
    assert.equal(elements.coverage.children.length, 4); assert.equal(elements.points.children.length, 8);
    assert.equal(elements.view.children.length, 3); assert.equal(elements.plot.children.filter(node => node.tag === "circle").length, 16);
    const upper = elements.plot.children.find(node => node.tag === "circle" && node.attrs["stroke-dasharray"] === "2 2");
    upper.events.focus(); assert.match(elements.detail.textContent, /lower 40\.00%; upper 90\.00%/);
    elements.view.value = "12"; elements.view.events.change();
    assert.equal(elements.points.children.length, 8); assert.ok(elements.plot.children.filter(node => node.tag === "circle").every(node => node.attrs.stroke === "none"));
    elements.axis.value = "steps"; elements.axis.events.change();
    assert.match(elements.plot.children.find(node => node.tag === "text" && node.textContent.includes("Training agent")).textContent, /decisions/);
    const impala = elements.legend.children[2].children[0]; impala.checked = false; impala.events.change();
    assert.equal(elements.points.children.length, 6); assert.equal(elements.coverage.children.length, 4);
});
test("actual empty application plots zero observations and labels all missing model costs", () => {
    const data = fixture(); data.status = "incomplete"; data.points = []; data.means = [];
    data.conditions[0].status = "incomplete"; data.conditions[0].observed = 0;
    data.missing = data.planned_cells.map(row => ({binding_index: row.binding_index, steps: row.steps}));
    const elements = application(data);
    assert.equal(elements.plot.children.filter(node => node.tag === "circle" || node.tag === "polyline").length, 0);
    assert.equal(elements.missing.children.length, 16); assert.equal(elements.costs.children.length, 8);
    assert.equal(elements.coverage.children[2].children[0].textContent, "IMPALA");
    assert.match(elements.status.textContent, /0\/16 observed; 16 missing/);
    assert.ok(elements.plot.children.some(node => node.textContent.includes("No audited policy observations")));
});
test("partial application shows remaining observations without certified rings or bridging a gap", () => {
    const data = fixture(); data.conditions[0].status = "incomplete";
    data.means = data.means.filter(row => row.model !== "flex_quality" || row.steps !== 100);
    const elements = application(data);
    assert.ok(elements.plot.children.filter(node => node.tag === "circle").every(node => node.attrs.stroke === "none"));
    assert.equal(elements.points.children.length, 7);
    elements.raw.checked = true; elements.raw.events.change();
    assert.ok(elements.plot.children.some(node => node.tag === "polyline" && node.attrs.opacity === .18));
});
