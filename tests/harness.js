// Uso: node harness.js <script.js> <snippet.js>. Imprime o JSON do ultimo valor do snippet.
const fs = require("fs"), vm = require("vm");
const handlers = {};
function fake(name) {
  const store = {};
  const target = function () {};
  const p = new Proxy(target, {
    get(_, k) {
      if (k === Symbol.toPrimitive) return () => "";
      if (k === "toString" || k === "valueOf") return () => "";
      if (k === "then") return undefined;
      if (k === "length") return 0;
      if (k === "value" || k === "innerHTML" || k === "textContent") return k in store ? store[k] : "";
      if (k === "dataset") return store[k] || (store[k] = {});
      if (k === "style") return store[k] || (store[k] = { setProperty() {}, removeProperty() {} });
      if (k === "disabled") return !!store[k];
      if (k === "getBoundingClientRect") return () => ({ width: 1000, height: 600, left: 0, top: 0, right: 1000, bottom: 600 });
      if (k === "querySelectorAll") return () => [];
      if (k === "addEventListener") return (ev, fn) => { (store.ev = store.ev || {})[ev] = fn; };
      if (k === "click") return () => { const f = (store.ev || {}).click; if (f) f({ target: p, preventDefault() {} }); };
      if (k === "getAttribute") return () => "";
      if (k === "classList") return { add() {}, remove() {}, toggle() {}, contains() { return false; } };
      if (k in store) return store[k];
      return fake(name + "." + String(k));
    },
    set(_, k, v) { store[k] = v; return true; },
    apply() { return fake(name + "()"); },
  });
  return p;
}
const els = {};
const doc = new Proxy({}, {
  get(_, k) {
    if (k === "getElementById") return (id) => els[id] || (els[id] = fake(id));
    if (k === "createElement" || k === "createElementNS") return () => fake("el");
    if (k === "querySelectorAll") return () => [];
    if (k === "querySelector") return () => fake("q");
    if (k === "addEventListener") return () => {};
    if (k === "documentElement") return fake("html");
    if (k === "activeElement") return null;
    return fake("document." + String(k));
  },
});
const mem = {};
const sandbox = {
  document: doc,
  sessionStorage: { getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = String(v); } },
  Plotly: new Proxy({}, { get: () => () => {} }),
  ResizeObserver: class { observe() {} },
  setTimeout: () => 0, console,
  getComputedStyle: () => ({ getPropertyValue: () => "" }),
  requestAnimationFrame: () => 0,
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  __els: els,
};
sandbox.window = sandbox;
sandbox.localStorage = sandbox.sessionStorage;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
const out = vm.runInContext(fs.readFileSync(process.argv[3], "utf8"), sandbox);
console.log(JSON.stringify(out === undefined ? null : out));
