(()=>{'use strict';
const input=document.querySelector('#audioInput'),deck=document.querySelector('.deck');if(!input||!deck)return;
const audioLike=file=>file&&(String(file.type||'').startsWith('audio/')||/\.(wav|mp3|m4a|aac|ogg|oga|flac|webm|opus)$/i.test(file.name||''));
function load(file){if(!audioLike(file))return false;const dt=new DataTransfer();dt.items.add(file);input.files=dt.files;input.dispatchEvent(new Event('change',{bubbles:true}));return true}
const sliderLabels={drive:'Input gain',bass:'Low EQ',presence:'Presence EQ',air:'High EQ',ceiling:'Export peak ceiling'};for(const[id,label]of Object.entries(sliderLabels)){const el=document.getElementById(id);if(el&&!el.getAttribute('aria-label'))el.setAttribute('aria-label',label)}
let dragDepth=0;
deck.addEventListener('dragenter',e=>{if(![...e.dataTransfer?.items||[]].some(x=>x.kind==='file'))return;e.preventDefault();dragDepth++;deck.classList.add('is-drop-target')});
deck.addEventListener('dragover',e=>{if(![...e.dataTransfer?.items||[]].some(x=>x.kind==='file'))return;e.preventDefault();if(e.dataTransfer)e.dataTransfer.dropEffect='copy';deck.classList.add('is-drop-target')});
deck.addEventListener('dragleave',()=>{dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)deck.classList.remove('is-drop-target')});
deck.addEventListener('drop',e=>{e.preventDefault();dragDepth=0;deck.classList.remove('is-drop-target');const file=[...e.dataTransfer?.files||[]].find(audioLike);if(file)load(file)});
document.addEventListener('paste',e=>{const el=e.target,tag=el?.tagName;if(el?.isContentEditable||['INPUT','TEXTAREA','SELECT'].includes(tag))return;const file=[...e.clipboardData?.files||[]].find(audioLike);if(file&&load(file))e.preventDefault()});
})();