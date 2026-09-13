import puppeteer from 'puppeteer-core';
import { access } from 'node:fs/promises';

const candidates=[process.env.CHROME_BIN,'/usr/bin/google-chrome','/usr/bin/google-chrome-stable','/usr/bin/chromium','/usr/bin/chromium-browser'].filter(Boolean);
let executablePath=null;for(const path of candidates){try{await access(path);executablePath=path;break}catch{}}
if(!executablePath)throw new Error('No Chrome/Chromium executable found');
const browser=await puppeteer.launch({executablePath,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
const page=await browser.newPage();
const base=process.env.SMOKE_BASE||'http://127.0.0.1:4173';
const errors=[];page.on('pageerror',e=>errors.push(String(e?.stack||e)));page.on('console',m=>{if(m.type()==='error'&&!/googletagmanager|Failed to load resource/i.test(m.text()))errors.push(m.text())});
try{
  const r=await page.goto(`${base}/pv-sites/image-shrink-jp/`,{waitUntil:'domcontentloaded',timeout:20000});if(!r?.ok())throw new Error(`HTTP ${r?.status()}`);
  for(const s of ['#fileInput','#batchQueue','#processAll','#downloadAll','#targetKb','#preview'])await page.waitForSelector(s);
  const multiple=await page.$eval('#fileInput',e=>e.multiple);if(!multiple)throw new Error('file input is not multiple');
  await page.evaluate(async()=>{const make=async(name,color)=>{const c=document.createElement('canvas');c.width=32;c.height=24;const x=c.getContext('2d');x.fillStyle=color;x.fillRect(0,0,c.width,c.height);x.fillStyle='#fff';x.fillRect(4,4,8,8);const blob=await new Promise(resolve=>c.toBlob(resolve,'image/png'));return new File([blob],name,{type:'image/png'})};const dt=new DataTransfer();dt.items.add(await make('one.png','#c33'));dt.items.add(await make('two.png','#36c'));const input=document.querySelector('#fileInput');input.files=dt.files;input.dispatchEvent(new Event('change',{bubbles:true}))});
  await page.waitForFunction(()=>document.querySelectorAll('#batchQueue .batch-item').length===2,{timeout:5000});
  await page.click('#processAll');
  await page.waitForFunction(()=>!document.querySelector('#downloadAll')?.disabled,{timeout:15000});
  const ready=await page.$eval('#batchSummary',e=>e.textContent);if(!/2 ready/.test(ready))throw new Error(`unexpected batch summary: ${ready}`);
  await page.evaluate(()=>{window.__zipBlob=null;window.__downloads=[];const create=URL.createObjectURL.bind(URL),click=HTMLAnchorElement.prototype.click;URL.createObjectURL=value=>{if(value instanceof Blob&&value.type==='application/zip')window.__zipBlob=value;return create(value)};HTMLAnchorElement.prototype.click=function(){if(this.download){window.__downloads.push(this.download);return}return click.call(this)}});
  await page.click('#downloadAll');
  await page.waitForFunction(()=>window.__downloads?.some(x=>/\.zip$/i.test(x))&&window.__zipBlob,{timeout:10000});
  const zip=await page.evaluate(async()=>{const a=new Uint8Array(await window.__zipBlob.arrayBuffer());return{size:a.length,magic:[...a.slice(0,4)]}});if(zip.size<100||zip.magic.join(',')!=='80,75,3,4')throw new Error(`invalid ZIP: ${JSON.stringify(zip)}`);
  if(errors.length)throw new Error(errors.join('\n'));
  console.log('PASS image batch compression + ZIP');
}finally{await page.close();await browser.close()}
