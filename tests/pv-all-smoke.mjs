import puppeteer from 'puppeteer-core';
import { access, readdir, stat } from 'node:fs/promises';
import path from 'node:path';

const candidates=[process.env.CHROME_BIN,'/usr/bin/google-chrome','/usr/bin/google-chrome-stable','/usr/bin/chromium','/usr/bin/chromium-browser'].filter(Boolean);
let executablePath=null;
for(const p of candidates){try{await access(p);executablePath=p;break}catch{}}
if(!executablePath)throw new Error('No Chrome/Chromium executable found');

const root=path.resolve(process.cwd(),'pv-sites');
const base=process.env.SMOKE_BASE||'http://127.0.0.1:4173';
const baseOrigin=new URL(base).origin;
const dirs=[];
for(const name of await readdir(root)){
  if(name.startsWith('.'))continue;
  const dir=path.join(root,name);
  try{if(!(await stat(dir)).isDirectory())continue;await access(path.join(dir,'index.html'));dirs.push(name)}catch{}
}
dirs.sort();
if(!dirs.length)throw new Error('No pv-sites applications found');

const browser=await puppeteer.launch({executablePath,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
const failures=[];
const ignoreConsole=/Failed to load resource|googletagmanager|google-analytics|profitableratecpmnetwork|ERR_BLOCKED_BY_CLIENT|favicon\.ico/i;
const local=url=>{try{return new URL(url).origin===baseOrigin}catch{return false}};

async function audit(name,viewport){
  const page=await browser.newPage();
  await page.setViewport(viewport);
  const errors=[];
  page.on('pageerror',e=>errors.push(`pageerror: ${e?.stack||e}`));
  page.on('requestfailed',req=>{const u=req.url();if(local(u)&&!/favicon\.ico(?:\?|$)/i.test(u))errors.push(`request failed: ${u} — ${req.failure()?.errorText||'unknown'}`)});
  page.on('response',res=>{const u=res.url(),status=res.status();if(local(u)&&status>=400&&!/favicon\.ico(?:\?|$)/i.test(u))errors.push(`HTTP ${status}: ${u}`)});
  page.on('console',msg=>{const text=msg.text();if(msg.type()==='error'&&!ignoreConsole.test(text))errors.push(`console: ${text}`)});
  try{
    const response=await page.goto(`${base}/pv-sites/${name}/`,{waitUntil:'domcontentloaded',timeout:20000});
    const status=response?.status()||0;if(!(status>=200&&status<400))throw new Error(`HTTP ${status}`);
    await new Promise(r=>setTimeout(r,700));
    const report=await page.evaluate(()=>{
      const visible=el=>{const s=getComputedStyle(el),r=el.getBoundingClientRect();return s.display!=='none'&&s.visibility!=='hidden'&&r.width>0&&r.height>0};
      const labelFor=el=>{
        if(el.getAttribute('aria-label')?.trim()||el.getAttribute('aria-labelledby')?.trim()||el.getAttribute('title')?.trim())return true;
        if(el.id&&document.querySelector(`label[for="${CSS.escape(el.id)}"]`))return true;
        return !!el.closest('label');
      };
      const unnamedButtons=[...document.querySelectorAll('button')].filter(visible).filter(b=>!(b.textContent||'').trim()&&!b.getAttribute('aria-label')&&!b.getAttribute('title')).length;
      const unlabeledInputs=[...document.querySelectorAll('input:not([type="hidden"]):not([type="file"]),select,textarea')].filter(visible).filter(el=>!labelFor(el)).length;
      const ids=[...document.querySelectorAll('[id]')].map(x=>x.id),dupes=ids.filter((id,i)=>ids.indexOf(id)!==i);
      const title=(document.title||'').trim(),description=document.querySelector('meta[name="description"]')?.content?.trim()||'',canonical=document.querySelector('link[rel="canonical"]')?.href||'';
      const body=document.body.getBoundingClientRect(),overflow=Math.max(0,document.documentElement.scrollWidth-document.documentElement.clientWidth);
      const h1=[...document.querySelectorAll('h1')].filter(visible).length;
      return{title,description,canonical,dupes:[...new Set(dupes)],unnamedButtons,unlabeledInputs,bodyW:body.width,bodyH:body.height,overflow,h1};
    });
    if(!report.title)errors.push('missing document title');
    if(report.description.length<30)errors.push(`meta description too short (${report.description.length})`);
    if(!/^https?:\/\//.test(report.canonical))errors.push('missing/invalid canonical URL');
    if(report.dupes.length)errors.push(`duplicate IDs: ${report.dupes.join(', ')}`);
    if(report.unnamedButtons)errors.push(`${report.unnamedButtons} visible button(s) have no accessible name`);
    if(report.unlabeledInputs)errors.push(`${report.unlabeledInputs} visible form control(s) have no accessible label`);
    if(report.bodyW<100||report.bodyH<100)errors.push('body did not render meaningful content');
    if(report.overflow>24)errors.push(`horizontal overflow: ${report.overflow}px`);
    if(report.h1===0)errors.push('no visible H1');
    if(errors.length)throw new Error(errors.join('\n'));
    console.log(`PASS ${name} ${viewport.width}x${viewport.height}`);
  }catch(e){failures.push(`${name} ${viewport.width}x${viewport.height}: ${e?.stack||e}`);console.error(`FAIL ${name} ${viewport.width}x${viewport.height}`,e)}finally{await page.close()}
}

for(const name of dirs)await audit(name,{width:1440,height:1000,deviceScaleFactor:1});
for(const name of dirs)await audit(name,{width:390,height:844,deviceScaleFactor:1,isMobile:true,hasTouch:true});
await browser.close();

console.log(`Audited ${dirs.length} tools across desktop and mobile (${dirs.length*2} page boots).`);
if(failures.length){console.error(`\nAll-tool quality failures (${failures.length}):\n${failures.join('\n\n')}`);process.exit(1)}
console.log('All pv-sites passed the production browser quality gate.');