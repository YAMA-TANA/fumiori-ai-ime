#!/usr/bin/env node
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const roots = ['pv-sites', 'sales-sites']
  .map(name => path.resolve(process.cwd(), name))
  .filter(dir => fs.existsSync(dir));
const errors = [];
const warnings = [];
const canonicalOwners = new Map();
let htmlCount = 0;
let jsCount = 0;
let inlineCount = 0;
let siteCount = 0;

function walk(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(p));
    else out.push(p);
  }
  return out;
}

function rel(p) { return path.relative(process.cwd(), p).replaceAll(path.sep, '/'); }
function fail(file, msg) { errors.push(`${rel(file)}: ${msg}`); }
function warn(file, msg) { warnings.push(`${rel(file)}: ${msg}`); }

function checkJs(file, source = null, label = '') {
  jsCount++;
  let target = file;
  let cleanup = false;
  if (source != null) {
    const safe = rel(file).replace(/[^a-z0-9_.-]+/gi, '_');
    target = path.join(os.tmpdir(), `static-tool-${process.pid}-${safe}-${inlineCount++}.js`);
    fs.writeFileSync(target, source, 'utf8');
    cleanup = true;
  }
  const r = spawnSync(process.execPath, ['--check', target], { encoding: 'utf8' });
  if (cleanup) fs.rmSync(target, { force: true });
  if (r.status !== 0) fail(file, `${label || 'JavaScript'} syntax error: ${(r.stderr || r.stdout).trim().split('\n').slice(-2).join(' ')}`);
}

function attrs(tag) {
  const out = new Map();
  const re = /([:\w-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g;
  let m;
  while ((m = re.exec(tag))) out.set(m[1].toLowerCase(), m[2] ?? m[3] ?? m[4] ?? '');
  return out;
}

function checkLocalRef(file, value, kind) {
  if (!value || /^(?:https?:|data:|blob:|mailto:|tel:|#|\/\/)/i.test(value)) return;
  const clean = value.split(/[?#]/)[0];
  if (!clean || clean.startsWith('/')) return;
  const target = path.resolve(path.dirname(file), clean);
  if (!fs.existsSync(target)) fail(file, `missing local ${kind}: ${value}`);
}

function metaValue(source, key, attrName = 'name') {
  for (const m of source.matchAll(/<meta\b[^>]*>/gi)) {
    const a = attrs(m[0]);
    if ((a.get(attrName) || '').toLowerCase() === key.toLowerCase()) return a.get('content') || '';
  }
  return '';
}

function linkValue(source, relation) {
  for (const m of source.matchAll(/<link\b[^>]*>/gi)) {
    const a = attrs(m[0]);
    const rels = (a.get('rel') || '').toLowerCase().split(/\s+/);
    if (rels.includes(relation.toLowerCase())) return a.get('href') || '';
  }
  return '';
}

function visibleTextLength(html) {
  return html
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&[a-z0-9#]+;/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim().length;
}

function checkSeoEntrypoint(file, source) {
  const title = source.match(/<title>([^<]+)<\/title>/i)?.[1]?.trim() || '';
  const description = metaValue(source, 'description');
  const robots = metaValue(source, 'robots');
  const canonical = linkValue(source, 'canonical');
  const ogTitle = metaValue(source, 'og:title', 'property');
  const ogDescription = metaValue(source, 'og:description', 'property');
  const ogUrl = metaValue(source, 'og:url', 'property');
  const h1Count = [...source.matchAll(/<h1\b/gi)].length;
  const jsonLdCount = [...source.matchAll(/<script\b[^>]*type=["']application\/ld\+json["'][^>]*>/gi)].length;
  const bodyChars = visibleTextLength(source);

  if (title.length < 12) warn(file, `title is unusually short (${title.length} chars)`);
  if (title.length > 78) warn(file, `title may truncate in search results (${title.length} chars)`);
  if (description.length < 55) warn(file, `meta description is short (${description.length} chars)`);
  if (description.length > 190) warn(file, `meta description is long (${description.length} chars)`);
  if (!robots.toLowerCase().includes('index') || !robots.toLowerCase().includes('follow')) fail(file, 'robots meta must include index,follow');
  if (!/^https:\/\//i.test(canonical)) fail(file, 'canonical must be an absolute HTTPS URL');
  if (canonical) {
    const other = canonicalOwners.get(canonical);
    if (other && other !== rel(file)) fail(file, `duplicate canonical also used by ${other}`);
    canonicalOwners.set(canonical, rel(file));
  }
  if (h1Count !== 1) fail(file, `expected exactly one h1, found ${h1Count}`);
  if (!ogTitle) fail(file, 'missing og:title');
  if (!ogDescription) fail(file, 'missing og:description');
  if (!ogUrl) fail(file, 'missing og:url');
  if (canonical && ogUrl && canonical !== ogUrl) fail(file, `og:url must match canonical (${canonical})`);
  if (jsonLdCount === 0) fail(file, 'missing JSON-LD structured data');
  if (bodyChars < 450) warn(file, `thin visible content (${bodyChars} characters)`);

  const dir = path.dirname(file);
  const robotsFile = path.join(dir, 'robots.txt');
  const sitemapFile = path.join(dir, 'sitemap.xml');
  const llmsFile = path.join(dir, 'llms.txt');
  if (!fs.existsSync(robotsFile)) fail(file, 'missing robots.txt');
  else {
    const r = fs.readFileSync(robotsFile, 'utf8');
    if (!/^User-agent:\s*\*/im.test(r)) fail(robotsFile, 'missing User-agent: *');
    if (!/^Allow:\s*\//im.test(r)) fail(robotsFile, 'missing Allow: /');
    if (!/^Sitemap:\s*https:\/\//im.test(r)) fail(robotsFile, 'missing absolute Sitemap URL');
  }
  if (!fs.existsSync(sitemapFile)) fail(file, 'missing sitemap.xml');
  else {
    const map = fs.readFileSync(sitemapFile, 'utf8');
    if (canonical && !map.includes(`<loc>${canonical}</loc>`)) fail(sitemapFile, `sitemap does not contain canonical ${canonical}`);
    if (!/<lastmod>\d{4}-\d{2}-\d{2}<\/lastmod>/i.test(map)) fail(sitemapFile, 'sitemap URLs should include lastmod');
  }
  if (!fs.existsSync(llmsFile)) warn(file, 'missing llms.txt discovery summary');
}

function checkHtml(file) {
  htmlCount++;
  const s = fs.readFileSync(file, 'utf8');
  if (!/^\s*<!doctype html>/i.test(s)) fail(file, 'missing <!doctype html>');
  if (!/<html\b[^>]*\blang\s*=/i.test(s)) fail(file, 'missing html lang');
  if (!/<meta\b[^>]*name=["']viewport["']/i.test(s)) fail(file, 'missing viewport meta');
  if (!/<title>[^<]+<\/title>/i.test(s)) fail(file, 'missing non-empty title');
  if (!/<meta\b[^>]*name=["']description["'][^>]*content=["'][^"']+/i.test(s) && !/<meta\b[^>]*content=["'][^"']+["'][^>]*name=["']description["']/i.test(s)) fail(file, 'missing meta description');
  if (!/<link\b[^>]*rel=["']canonical["'][^>]*href=/i.test(s) && !/<link\b[^>]*href=[^>]*rel=["']canonical["']/i.test(s)) warn(file, 'missing canonical link');

  const ids = new Map();
  for (const m of s.matchAll(/\bid\s*=\s*["']([^"']+)["']/gi)) ids.set(m[1], (ids.get(m[1]) || 0) + 1);
  for (const [id, count] of ids) if (count > 1) fail(file, `duplicate id="${id}" (${count} occurrences)`);

  for (const m of s.matchAll(/<script\b[^>]*>[\s\S]*?<\/script>/gi)) {
    const full = m[0];
    const open = full.match(/^<script\b[^>]*>/i)?.[0] || '<script>';
    const a = attrs(open);
    const src = a.get('src');
    if (src) { checkLocalRef(file, src, 'script'); continue; }
    const type = (a.get('type') || '').toLowerCase();
    if (type && !['text/javascript', 'application/javascript', 'module'].includes(type)) continue;
    const body = full.slice(open.length, full.toLowerCase().lastIndexOf('</script>'));
    if (body.trim()) checkJs(file, body, 'inline JavaScript');
  }

  for (const m of s.matchAll(/<link\b[^>]*>/gi)) {
    const a = attrs(m[0]);
    const href = a.get('href');
    const relation = (a.get('rel') || '').toLowerCase();
    if (href && relation.includes('stylesheet')) checkLocalRef(file, href, 'stylesheet');
  }

  for (const m of s.matchAll(/<a\b[^>]*\btarget\s*=\s*["']_blank["'][^>]*>/gi)) {
    const a = attrs(m[0]);
    const r = (a.get('rel') || '').toLowerCase().split(/\s+/);
    if (!r.includes('noopener')) fail(file, 'target="_blank" link missing rel="noopener"');
  }

  for (const m of s.matchAll(/\son[a-z]+\s*=/gi)) warn(file, `inline event handler ${m[0].trim()} reduces CSP hardening options`);

  if (path.basename(file).toLowerCase() === 'index.html' && roots.some(root => path.dirname(file) !== root)) {
    siteCount++;
    checkSeoEntrypoint(file, s);
  }
}

if (!roots.length) {
  console.error('No marketing site roots found');
  process.exit(2);
}

for (const root of roots) {
  const files = walk(root);
  for (const file of files) {
    if (file.endsWith('.js')) checkJs(file);
    else if (file.endsWith('.html')) checkHtml(file);
  }
}

console.log(`Validated ${siteCount} marketing sites, ${htmlCount} HTML files, ${jsCount} JavaScript blocks/files.`);
if (warnings.length) {
  console.log(`\nWarnings (${warnings.length}):`);
  for (const w of warnings) console.log(`  - ${w}`);
}
if (errors.length) {
  console.error(`\nErrors (${errors.length}):`);
  for (const e of errors) console.error(`  - ${e}`);
  process.exit(1);
}
console.log('\nStatic marketing + SEO quality gate passed.');
