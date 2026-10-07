/* Offline display helpers. Stored means/flags are never recomputed here. */
"use strict";
const PixelFrontier = (() => {
    const models = ["flex_quality", "nature_cnn", "impala_cnn", "impoola_cnn"];
    const labels = {flex_quality: "Ours quality", nature_cnn: "Nature", impala_cnn: "IMPALA", impoola_cnn: "Impoola"};
    const colors = {flex_quality: "#1767c0", nature_cnn: "#b96600", impala_cnn: "#21834a", impoola_cnn: "#9735b0"};
    const aliases = {"nature-cnn": "nature_cnn", "impala-cnn": "impala_cnn", "impoola-cnn": "impoola_cnn"};
    function catalog(data) {
        // Allocation determines membership; a missing/losing model stays visible.
        const allocated = [...new Set((data.planned_cells || []).map(row => row.model))];
        if (data.model_catalog !== undefined && !Array.isArray(data.model_catalog)) throw new Error("Invalid model catalog type");
        const names = data.model_catalog === undefined ? allocated : [...data.model_catalog];
        if (names.some(name => typeof name !== "string" || !name) || new Set(names).size !== names.length
                || allocated.some(name => !names.includes(name)) || names.some(name => !allocated.includes(name)))
            throw new Error("Invalid declared model catalog");
        const namesSet = new Set(names), display = Object.create(null), palette = Object.create(null);
        if ([...(data.points || []), ...(data.means || [])].some(row => !namesSet.has(row.model)))
            throw new Error("Observed model is absent from allocation");
        for (const name of names) {
            const canonical = Object.hasOwn(aliases, name) ? aliases[name] : name;
            display[name] = Object.hasOwn(labels, canonical) ? labels[canonical] : name;
            let hash = 2166136261;
            for (const character of name) hash = Math.imul(hash ^ character.codePointAt(0), 16777619) >>> 0;
            palette[name] = Object.hasOwn(colors, canonical) ? colors[canonical] : `hsl(${hash % 360} 70% 38%)`;
        }
        return {models: names, labels: display, colors: palette};
    }
    const key = row => JSON.stringify([row.environment, row.training_appearance, row.evaluation_representation]);
    const cellKey = row => `${row.binding_index}:${row.steps}`;
    function select(data, condition, view, visible = catalog(data).models) {
        const matches = row => key(row) === key(condition);
        const planned = data.planned_cells.filter(matches);
        const observations = data.points.filter(matches);
        const means = data.means.filter(matches);
        const points = (view === "mean" ? means : observations.filter(row => row.seed === Number(view)))
            .filter(row => visible.includes(row.model));
        const observed = new Set(observations.map(cellKey));
        const jobs = new Set(planned.map(row => row.job_index));
        return {planned, observations, means, points,
            seeds: [...new Set(planned.map(row => row.seed))].sort((a, b) => a - b),
            missing: planned.filter(row => !observed.has(cellKey(row))),
            failures: data.failures.filter(row => jobs.has(row.job_index) || planned.some(cell => cell.binding_index === row.binding_index)),
            costs: data.training_job_costs.filter(row => jobs.has(row.job_index))};
    }
    function segments(rows, expectedSteps) {
        const order = [...new Set(expectedSteps)].sort((a, b) => a - b);
        const positions = new Map(order.map((step, index) => [step, index]));
        const ordered = [...rows].sort((a, b) => a.steps - b.steps);
        const result = [];
        for (const row of ordered) {
            const previous = result.length ? result[result.length - 1].at(-1) : null;
            if (!previous || positions.get(row.steps) !== positions.get(previous.steps) + 1) result.push([]);
            result[result.length - 1].push(row);
        }
        return result;
    }
    function marked(row, condition, view, axis, bound) {
        return condition.status === "complete_descriptive" && view === "mean" && axis === "seconds"
            && row[`frontier_seconds_${bound}`] === true;
    }
    function bounds(points, axis, fraction) {
        let xmax = 1, ymax = 1;
        for (const row of points) {xmax = Math.max(xmax, row[axis]); ymax = Math.max(ymax, row.score_upper);}
        return {xmax: xmax * 1.05, ymax: fraction ? 1 : ymax * 1.05};
    }
    return {models, labels, colors, catalog, key, cellKey, select, segments, marked, bounds};
})();
if (typeof module !== "undefined") module.exports = PixelFrontier;
