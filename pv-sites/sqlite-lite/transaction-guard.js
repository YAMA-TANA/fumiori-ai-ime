(()=>{'use strict';
const $=s=>document.querySelector(s),states=new Set(['トランザクション中','Transaction active','事务进行中','트랜잭션 진행 중']);
const messages={ja:'トランザクション中はDB保存できません。COMMITまたはROLLBACKしてから保存してください。',en:'Database export is disabled during an active transaction. COMMIT or ROLLBACK first.',zh:'事务进行中时无法保存数据库。请先 COMMIT 或 ROLLBACK。',ko:'트랜잭션 진행 중에는 DB를 저장할 수 없습니다. 먼저 COMMIT 또는 ROLLBACK 하세요.'};
function lang(){return $('#languageSelect')?.value||document.documentElement.lang||'en'}
function active(){return states.has(($('#dbState')?.textContent||'').trim())}
function sync(){const b=$('#exportDb');if(!b)return;b.dataset.transactionBlocked=active()?'true':'false';b.title=active()?(messages[lang()]||messages.en):''}
function init(){const b=$('#exportDb'),state=$('#dbState');if(!b||!state)return;b.addEventListener('click',e=>{if(!active())return;e.preventDefault();e.stopImmediatePropagation();const s=$('#status');if(s)s.textContent=messages[lang()]||messages.en},{capture:true});new MutationObserver(sync).observe(state,{childList:true,subtree:true,characterData:true});$('#languageSelect')?.addEventListener('change',()=>setTimeout(sync));sync()}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})();