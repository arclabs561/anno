import init, {extract} from './engine/pkg/saltglass_reader.js';

const $ = selector => document.querySelector(selector);
const log = "From the landing field at Saltglass Reach you can usually see two of the three moons. To find the third, stand beside the mail shed and look through the gap in the observatory stairs. Iona Vale painted a white mark on the step for visitors. She touches it up every spring.\n\nVale captains the Kestrel for the Brackwater Packet Company. Yesterday she brought the company’s new tide tables over from Low Bell, along with one passenger and a crate of dry socks. The tables say the mail road stays dry until noon. The passenger walked in at eleven with his boots full of water.";
const catalog = {
  registry: ['Port gazetteer', 0, 'Anno LexiconNER looks up local names in the Saltglass registry. The vessel register adds the Kestrel to its vocabulary.'],
  heuristic: ['Heuristic reader', 1, 'Anno HeuristicNER uses capitalization, nearby words and other cues. Try names and organizations from beyond the Reach.'],
  patterns: ['Pattern reader', 4, 'Anno RegexNER recognizes structured text. Try a date, a sum of money or a (fictional) email address in the reader lab.'],
  vessels: ['Vessel register', 6, 'The harbor supplement lists ships by name. Fit it alongside the port gazetteer to identify the Kestrel in the dispatch.']
};
const readers = ['registry','heuristic','patterns'];
const key = 'anno.saltglass.v1';
let saved;
try { saved = JSON.parse(localStorage.getItem(key)); } catch {}
const state = {
  reader: readers.includes(saved?.reader) ? saved.reader : 'registry',
  vessels: saved?.vessels === true,
  mission: ['accepted','delivered'].includes(saved?.mission) ? saved.mission : 'open',
  sample: typeof saved?.sample === 'string' ? saved.sample.slice(0,6000) : log,
  away: saved?.away === true
};
let screen = 'log', selected = state.reader, ready = false, labResult = null;
function save() { try { localStorage.setItem(key, JSON.stringify(state)); } catch {} }
function el(tag,text,cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function action(name,fn) { const node=el('button',name); node.onclick=fn; return node; }
function read(text,reader=state.reader,vessels=state.vessels) {
  return JSON.parse(extract(text,reader,vessels));
}
function category(label) {
  return ({PER:'person',PERSON:'person',ORG:'organization',ORGANIZATION:'organization',
    LOC:'place',LOCATION:'place',GPE:'place',VESSEL:'vessel'})[label.toUpperCase()] || label.toLowerCase();
}
function annotated(text,entities) {
  const part=el('div'); const chars=Array.from(text);
  let paragraph=el('p'), cursor=0;
  part.append(paragraph);
  function plain(string) {
    string.split('\n\n').forEach((piece,i)=>{
      if(i) {paragraph=el('p');part.append(paragraph);}
      paragraph.append(piece);
    });
  }
  // Nested entities still appear in the JSON export. Draw the longest span first.
  for(const e of [...entities].sort((a,b)=>a.start-b.start || b.end-a.end)) {
    if(e.start<cursor || e.end>chars.length || e.end<=e.start) continue;
    plain(chars.slice(cursor,e.start).join(''));
    const type=category(e.label);
    const mark=el('mark',chars.slice(e.start,e.end).join(''),type);
    mark.title=type;mark.setAttribute('aria-label',mark.textContent+', '+type);
    paragraph.append(mark);cursor=e.end;
  }
  plain(chars.slice(cursor).join(''));
  const legend=el('div');
  for(const t of new Set(entities.map(e=>category(e.label)))) legend.append(el('span',t,'key '+t));
  part.append(legend);
  return part;
}
function draw() {
  const content=$('#content');content.replaceChildren();
  $('#port').hidden=state.away;$('#orbit').hidden=!state.away;
  $('#orbit-message').textContent=state.mission==='delivered'?'The Kestrel clears the flats with the corrected tide tables.':'A lamp burns above the mail shed. The port is accepting arrivals.';
  $('#heading').textContent=({log:'Saltglass Reach',lab:'Reader lab',board:'Mission BBS'})[screen];
  document.querySelectorAll('[data-screen]').forEach(b=>b.setAttribute('aria-current',String(b.dataset.screen===screen)));
  $('#export').disabled=!ready;
  if(screen==='log') content.append(annotated(log,ready?read(log):[]));
  if(screen==='lab') {
    const input=el('textarea');input.id='sample';input.rows=5;input.maxLength=6000;input.value=state.sample;
    const caption=el('label','Local text · '+catalog[state.reader][0]);caption.htmlFor=input.id;
    const result=el('div');result.setAttribute('aria-live','polite');
    const run=()=>{
      try {labResult=read(input.value);result.replaceChildren(el('p',labResult.length+' mentions','small'),annotated(input.value,labResult));}
      catch {result.textContent='The reader could not process this passage.';}
    };
    input.oninput=()=>{state.sample=input.value;labResult=null;result.replaceChildren();save();};
    const scan=action('Run Anno',run);scan.disabled=!ready;
    content.append(caption,input,scan,result);
    if(ready && labResult!==null) run();
  }
  if(screen==='board') {
    content.append(el('h3','Mail run · corrected tide tables'));
    if(state.mission==='open') content.append(el('p','The road flooded an hour early. Get the corrected tables to the packet boat before the next tide.'),action('Accept dispatch',()=>{state.mission='accepted';save();draw();}));
    if(state.mission==='accepted') {
      const dispatch='Iona Vale is waiting at Low Bell. Hand the corrected tide tables to the captain of the Kestrel.';
      const spans=ready?read(dispatch):[];
      content.append(annotated(dispatch,spans));
      const ship=spans.find(e=>category(e.label)==='vessel');
      if(ship) content.append(action('Deliver to '+ship.text,()=>{state.mission='delivered';save();draw();}));
      else content.append(el('p','Fit the port gazetteer and vessel register in the Outfitter to identify the ship.','small'));
    }
    if(state.mission==='delivered') content.append(el('p','Dispatch signed by Iona Vale. The Kestrel carries the revised tables to Low Bell. Your return passage is paid.'));
  }
}
function isFitted(id) {return id==='vessels'?state.vessels:state.reader===id;}
function sprite(node,i){node.style.backgroundPosition=(i%4)*100/3+'% '+Math.floor(i/4)*50+'%';}
function shop(){
  const inv=$('#inventory');inv.replaceChildren();
  for(const [id,[name,index]] of Object.entries(catalog)){
    const tile=action('',()=>{selected=id;shop();$('#tile-'+id).focus();});tile.id='tile-'+id;tile.className='tile';
    tile.setAttribute('aria-pressed',String(id===selected));
    const icon=el('span',undefined,'sprite');sprite(icon,index);
    tile.append(icon,name);inv.append(tile);
  }
  const [name,index,desc]=catalog[selected];$('#item-name').textContent=name;$('#description').textContent=desc;
  sprite($('#large-icon'),index);
  $('#equipment').textContent=isFitted(selected)?'Fitted':'Available';
  $('#fit').disabled=isFitted(selected);$('#remove').disabled=selected!=='vessels'||!state.vessels;
  const example='Vale captains the Kestrel for the Brackwater Packet Company.';
  const preview=$('#preview');preview.replaceChildren();
  if(ready) preview.append(el('span','Fitting preview','small'),annotated(example,read(example,selected==='vessels'?'registry':selected,selected==='vessels'||state.vessels)));
}
function updateFit(){
  labResult=null;save();shop();draw();
  $('#signal').textContent='Anno WebAssembly · '+catalog[state.reader][0];
}
$('#fit').onclick=()=>{if(selected==='vessels'){state.vessels=true;state.reader='registry';}else state.reader=selected;updateFit();};
$('#remove').onclick=()=>{state.vessels=false;updateFit();};
$('#shop').onclick=()=>{selected=state.reader;shop();$('#outfitter').showModal();};
$('#done').onclick=()=>$('#outfitter').close();
document.querySelectorAll('[data-screen]').forEach(b=>b.onclick=()=>{screen=b.dataset.screen;draw();$('#heading').focus();});
$('#leave').onclick=()=>{state.away=true;save();draw();$('#land').focus();};
$('#land').onclick=()=>{state.away=false;screen='log';save();draw();$('#heading').focus();};
$('#export').onclick=()=>{
  const text=screen==='lab'?state.sample:log;
  const blob=new Blob([JSON.stringify({reader:state.reader,vessel_register:state.vessels,text,entities:read(text)},null,2)],{type:'application/json'});
  const a=el('a');a.href=URL.createObjectURL(blob);a.download='saltglass-log.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};
draw();
try {await init();ready=true;draw();$('#signal').textContent='Anno WebAssembly · '+catalog[state.reader][0];}
catch {$('#signal').textContent='Reader unavailable. Reload to retry.';}
