// Time the span models in a real browser: onnxruntime-web on WebAssembly in headless Chromium, driven by Playwright.
//
//   node bench.mjs --models small=../../checkpoints/small-K-s0/onnx/model_int8.onnx,... [--threads 1,4]
//                  [--lengths 32,128,512] [--runs 30] [--out ../../results/latency_browser.json]
//
// A local server serves the page, onnxruntime-web and the model files. It sends the COOP/COEP headers, so the page is
// cross-origin isolated and the WebAssembly backend may use threads; --threads 1 measures the single-threaded case
// every browser supports. For each model and thread count the page records the session-creation time and, per input
// length, the median and 90th-percentile time of one forward pass (batch 1) over --runs timed runs after 5 warm-ups.
// Token ids are fixed pseudo-random ids, so the timing does not depend on the text.
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { createReadStream, existsSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { cpus, release, platform } from 'node:os';

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : fallback; };
const here = resolve(import.meta.dirname);
const models = Object.fromEntries(arg('models', '').split(',').filter(Boolean).map((kv) => kv.split('=')));
const threads = arg('threads', '1,4').split(',').map(Number);
const lengths = arg('lengths', '32,128,512').split(',').map(Number);
const runs = Number(arg('runs', '30'));
const out = arg('out', join(here, 'latency_browser.json'));
const vocab = Object.fromEntries(arg('vocab', '').split(',').filter(Boolean).map((kv) => kv.split('=').map((x, i) => (i ? Number(x) : x))));

const types = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream' };
const routes = { '/': join(here, 'page.html') };
const ortDir = join(here, 'node_modules', 'onnxruntime-web', 'dist');
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  let file = routes[url.pathname];
  if (!file && url.pathname.startsWith('/ort/')) file = join(ortDir, url.pathname.slice(5));
  if (!file && url.pathname.startsWith('/model/')) file = resolve(models[url.pathname.slice(7)] ?? '/nonexistent');
  if (!file || !existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, {
    'Content-Type': types[extname(file)] ?? 'application/octet-stream', 'Content-Length': statSync(file).size,
    'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp', 'Cache-Control': 'no-store',
  });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

// The headless shell Playwright installed for this version, or the one in the shared cache.
let executablePath;
const cache = `${process.env.HOME}/.cache/ms-playwright`;
const shell = existsSync(cache) ? readdirSync(cache).filter((d) => d.startsWith('chromium_headless_shell-')).sort().pop() : null;
if (shell) {
  const d = join(cache, shell);
  const sub = readdirSync(d).find((x) => x.startsWith('chrome-headless-shell'));
  if (sub) executablePath = join(d, sub, 'chrome-headless-shell');
}
const browser = await chromium.launch(executablePath ? { executablePath } : {});
const results = [];
for (const [name] of Object.entries(models)) {
  for (const t of threads) {
    const page = await (await browser.newContext()).newPage();
    const q = new URLSearchParams({ model: name, threads: String(t), lengths: lengths.join(','), runs: String(runs), vocab: String(vocab[name] ?? 30000) });
    await page.goto(`${base}/?${q}`);
    await page.waitForFunction(() => window.__done === true, null, { timeout: 30 * 60 * 1000, polling: 500 });
    const r = await page.evaluate(() => window.__result);
    results.push(r);
    console.log(JSON.stringify({ model: r.model, threads: r.threads, isolated: r.isolated, create_ms: r.create_ms, error: r.error, ...Object.fromEntries(r.lengths.map((l) => [`len${l.length}`, l.median_ms])) }));
    await page.context().close();
  }
}
const ua = await (await browser.newPage()).evaluate(() => navigator.userAgent);
writeFileSync(out, `${JSON.stringify({
  browser: `${browser.browserType().name()} ${browser.version()}`, userAgent: ua, onnxruntime_web: '1.30.0', backend: 'wasm',
  host: { cpu: cpus()[0]?.model, logical_cpus: cpus().length, os: `${platform()} ${release()}` }, runs, warmup: 5, results,
  finished: new Date().toISOString(),
}, null, 2)}\n`);
await browser.close();
server.close();
