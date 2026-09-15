// DOM smoke check of the standalone artifact; not a browser rendering test.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
class Element {
    constructor() { this.children = []; this.style = {}; this.value = ''; this.checked = false; }
    appendChild(child) { this.children.push(child); return child; }
    replaceChildren() { this.children = []; }
    setAttribute() {}
    addEventListener() {}
}
const ids = Object.fromEntries(['view','axis','zoom','ci','raw','legend','plot','detail','table'].map(k => [k, new Element()]));
ids.view.value = 'mean'; ids.axis.value = 'seconds'; ids.zoom.checked = ids.ci.checked = true;
const document = {getElementById: k => ids[k], createElement: () => new Element(),
    createElementNS: () => new Element(), createTextNode: text => ({textContent: text})};
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const context = vm.createContext({document});
vm.runInContext(script, context);
assert.equal(ids.legend.children.length, 6);
for (const view of ['mean','173','174','175','176','177']) {
    for (const axis of ['seconds','steps']) {
        for (const zoom of [false,true]) {
            for (const raw of [false,true]) {
                ids.view.value = view; ids.axis.value = axis;
                ids.zoom.checked = zoom; ids.raw.checked = raw;
                vm.runInContext('draw()', context);
                assert(ids.plot.children.length > 20);
                assert(ids.table.children.length > 0);
            }
        }
    }
}
vm.runInContext('visible.clear(); draw()', context);
assert.equal(ids.table.children.length, 0);
console.log('PASS: generated JavaScript, 48 chart views, six model controls and empty selection. Browser appearance not tested.');
