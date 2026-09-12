import puppeteer from 'puppeteer-core';
import { access } from 'node:fs/promises';

const candidates=[process.env.CHROME_BIN,'/usr/bin/google-chrome','/usr/bin/google-chrome-stable','/usr/bin/chromium','/usr/bin/chromium-browser'].filter(Boolean);
let executablePath=null;for(const path of candidates){try{await access(path);executablePath=path;break}catch{}}
if(!executablePath)throw new Error('No Chrome/Chromium executable found');
const browser=await puppeteer.launch({executablePath,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
const base=process.env.SMOKE_BASE||'http://127.0.0.1:4173',baseOrigin=new URL(base).origin,failures=[];
const local=url=>{try{return new URL(url).origin===baseOrigin}catch{return false}};

async function open(path,check){
  const page=await browser.newPage(),errors=[];
  page.on('pageerror',err=>errors.push(String(err?.stack||err)));
  page.on('response',response=>{const status=response.status(),url=response.url();if(local(url)&&status>=400&&!/\/favicon\.ico(?:\?|$)/i.test(url))errors.push(`HTTP ${status}: ${url}`)});
  page.on('requestfailed',request=>{const url=request.url();if(local(url)&&!/\/favicon\.ico(?:\?|$)/i.test(url))errors.push(`request failed: ${url} — ${request.failure()?.errorText||'unknown'}`)});
  page.on('console',msg=>{const text=msg.text();if(msg.type()==='error'&&!/Failed to load resource|googletagmanager|ERR_BLOCKED_BY_CLIENT/i.test(text))errors.push(`console: ${text}`)});
  try{const response=await page.goto(`${base}${path}`,{waitUntil:'domcontentloaded',timeout:20000});if(!response?.ok())throw new Error(`HTTP ${response?.status()} for ${path}`);await check(page);await new Promise(r=>setTimeout(r,500));if(errors.length)throw new Error(errors.join('\n'));console.log(`PASS ${path}`)}catch(error){failures.push(`${path}: ${error.stack||error}`);console.error(`FAIL ${path}`,error)}finally{await page.close()}
}

await open('/pv-sites/pixel-lite/',async page=>{await page.waitForSelector('#canvas');await page.waitForSelector('#layerList .layer-row');for(const s of ['#playAnimation','#frameList','#exportSheet'])if(!await page.$(s))throw new Error(`missing ${s}`);const before=await page.$$eval('#layerList .layer-row',r=>r.length);await page.click('#addLayer');const after=await page.$$eval('#layerList .layer-row',r=>r.length);if(after!==before+1)throw new Error(`layer add failed: ${before} -> ${after}`);await page.click('[data-tool="select"]');const box=await page.$eval('#canvas',el=>{const r=el.getBoundingClientRect();return{x:r.left,y:r.top,w:r.width,h:r.height}});await page.mouse.move(box.x+box.w*.2,box.y+box.h*.2);await page.mouse.down();await page.mouse.move(box.x+box.w*.45,box.y+box.h*.4,{steps:4});await page.mouse.up();const selection=await page.$eval('#selectionInfo',el=>el.textContent.trim());if(!selection||selection==='—')throw new Error('marquee selection did not initialize')});

await open('/pv-sites/audio-master-lite/',async page=>{await page.waitForSelector('#waveCanvas');for(const s of ['#trimSelection','#fadeIn','#fadeOut','#normalize','#undoEdit','#redoEdit','#playSelection','#exportSelection','#exportWav'])if(!await page.$(s))throw new Error(`missing ${s}`);if(!await page.$eval('#trimSelection',el=>el.disabled))throw new Error('trim should be disabled before audio is loaded')});

await open('/pv-sites/pdf-workbench/',async page=>{await page.waitForSelector('#pdfCanvas');await page.waitForFunction(()=>Boolean(window.pdfjsLib&&window.PDFLib),{timeout:20000});for(const s of ['[data-tool="select"]','[data-tool="whiteout"]','#exportTop','#undoTop','#redoTop'])if(!await page.$(s))throw new Error(`missing ${s}`);const label=await page.$eval('[data-tool="whiteout"]',el=>el.textContent.trim());if(!label)throw new Error('redaction control has no label')});

await open('/pv-sites/voice-meter/',async page=>{await page.waitForSelector('#waveCanvas');for(const s of ['#micButton','#startButton','#stopButton','#promptText','#scoreValue'])if(!await page.$(s))throw new Error(`missing ${s}`);const text=await page.$eval('#promptText',el=>el.value.trim());if(!text)throw new Error('practice prompt did not initialize')});

await open('/pv-sites/walk-air/',async page=>{for(const s of ['#cityForm','#cityInput','#locationButton','#dashboardCard','#sourceStatus']){await page.waitForSelector(s);if(!await page.$(s))throw new Error(`missing ${s}`)}});

await open('/pv-sites/nature-pulse/',async page=>{for(const s of ['#placeForm','#placeInput','#gpsButton','#observationGrid','#status']){await page.waitForSelector(s);if(!await page.$(s))throw new Error(`missing ${s}`)}});

await open('/pv-sites/food-lens/',async page=>{for(const s of ['#searchForm','#searchInput','#productGrid','#detailCard','#status']){await page.waitForSelector(s);if(!await page.$(s))throw new Error(`missing ${s}`)}});

await browser.close();if(failures.length){console.error('\nBrowser smoke failures:\n'+failures.join('\n\n'));process.exit(1)}
