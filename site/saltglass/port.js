import init, {extract, inspect} from './engine/pkg/saltglass_reader.js';

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
  sample: log,
  away: saved?.away === true
};
let screen = 'log', selected = state.reader, ready = false, labResult = null;
let experiment = 'mentions';
const dispatch = 'Iona Vale is waiting at Low Bell. Hand the corrected tide tables to the captain of the Kestrel.';
const experiments = {
  mentions:['Mentions', log],
  compare:['Compare readers', '🌙 Iona Vale reached Low Bell. Contact iona@example.org on May 12, 2026.'],
  links:['Linked mentions', 'Iona Vale sailed to Low Bell. Iona Vale returned to Saltglass Reach.'],
  relations:['Relation candidates', 'Iona Vale founded Brackwater Packet Company.'],
  redact:['Redact contacts', 'Reach the fictional harbor clerk at iona@example.org.']
};
function save() { try { localStorage.setItem(key, JSON.stringify({reader:state.reader,vessels:state.vessels,mission:state.mission,away:state.away})); } catch {} }
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
  const part=el('div'); const detail=el('output',undefined,'span-detail'); const chars=Array.from(text);
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
    const reveal=()=>{detail.textContent=e.text+' · '+type+' · characters ['+e.start+', '+e.end+')';};
    mark.tabIndex=0;mark.setAttribute('role','button');mark.onclick=reveal;
    mark.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();reveal();}};
    mark.title=type;mark.setAttribute('aria-label',mark.textContent+', '+type);
    paragraph.append(mark);cursor=e.end;
  }
  plain(chars.slice(cursor).join(''));
  const legend=el('div');
  for(const t of new Set(entities.map(e=>category(e.label)))) legend.append(el('span',t,'key '+t));
  part.append(legend,detail);
  return part;
}
function analyze(text) {
  if(experiment==='compare') {
    return {experiment,readers:Object.fromEntries(readers.map(id=>[id,read(text,id)]))};
  }
  if(experiment==='mentions') return {experiment,entities:read(text)};
  const value=inspect(text,state.reader,state.vessels,experiment);
  return experiment==='redact'?{experiment,redacted:value}:{experiment,groups:JSON.parse(value)};
}

function showAnalysis(result,text,report){
  result.replaceChildren();
  if(report.entities){
    result.append(el('p',report.entities.length+' mentions','small'),annotated(text,report.entities));
  } else if(report.readers){
    for(const [id,rows] of Object.entries(report.readers)){
      const group=el('details');group.open=true;
      group.append(el('summary',catalog[id][0]+' · '+rows.length),annotated(text,rows));
      result.append(group);
    }
  } else if(report.experiment==='redact'){
    result.append(el('p','Pattern-based redaction · review before sharing','small'),el('p',report.redacted));
  } else {
    const links=report.experiment==='links';
    result.append(el('p',links?'Rule-based links among detected mentions':'Heuristic candidates · founded, works for, located in','small'));
    if(!report.groups.length)result.append(el('p',links?'No repeated mentions linked.':'No matching relation found.'));
    for(const row of report.groups){
      result.append(el('p',links?row.map(m=>m.text+' ['+m.start+', '+m.end+')').join(' ↔ '):[row.head,row.predicate,row.tail].join(' → ')));
    }
  }
}
function draw() {
  const content=$('#content');content.replaceChildren();
  $('#port').hidden=state.away;$('#orbit').hidden=!state.away;
  $('#harbor-call').textContent = {open:'Mail road flooded · use west berth', accepted:'West berth · Kestrel awaiting dispatch', delivered:'Channel clear · Kestrel outbound'}[state.mission];
  $('#orbit-message').textContent=state.mission==='delivered'?'The Kestrel clears the flats with the corrected tide tables.':'A lamp burns above the mail shed. The port is accepting arrivals.';
  $('#heading').textContent=({log:'Saltglass Reach',lab:'Reader lab',board:'Mission BBS'})[screen];
  document.querySelectorAll('[data-screen]').forEach(b=>b.setAttribute('aria-current',String(b.dataset.screen===screen)));
  $('#export').disabled=!ready || (screen==='board' && state.mission!=='accepted');
  $('#export').textContent=screen==='board'?'Export dispatch':'Export log';
  if(screen==='log') content.append(annotated(log,ready?read(log):[]));
  if(screen==='lab') {
    const input=el('textarea');input.id='sample';input.rows=5;input.maxLength=6000;input.value=state.sample;
    const caption=el('label','Local text · '+catalog[state.reader][0]);caption.htmlFor=input.id;
    const result=el('div');result.setAttribute('aria-live','polite');
    const run=()=>{
      try {
        const measured=analyze(input.value);
        labResult=measured;
        showAnalysis(result,input.value,measured);
      } catch(error) {labResult=null;result.textContent=String(error);}
    };
    input.oninput=()=>{state.sample=input.value;labResult=null;result.replaceChildren();save();};
    const scan=action('Run Anno',run);scan.disabled=!ready;
    const selector=el('select');
    selector.id='experiment';
    for(const [value,[name]] of Object.entries(experiments)){
      const option=el('option',name);option.value=value;selector.append(option);
    }
    selector.value=experiment;
    selector.onchange=()=>{experiment=selector.value;labResult=null;result.replaceChildren();};
    const mode=el('label','Experiment ');mode.append(selector);
    const example=action('Load example',()=>{
      input.value=experiments[experiment][1]; input.oninput(); if(ready)run();
    });
    content.append(mode,caption,input,scan,example,result);
    if(ready && labResult!==null) run();
  }
  if(screen==='board') {
    content.append(el('h3','Mail run · corrected tide tables'));
    if(state.mission==='open') content.append(el('p','The road flooded an hour early. Get the corrected tables to the packet boat before the next tide.'),action('Accept dispatch',()=>{state.mission='accepted';save();draw();}));
    if(state.mission==='accepted') {
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
  $('#fit').disabled=!ready||isFitted(selected);$('#remove').disabled=!ready||selected!=='vessels'||!state.vessels;
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
  const text=screen==='lab'?state.sample:screen==='board'?dispatch:log;
  let report;
  try {
    report=screen==='lab'?analyze(text):{entities:read(text)};
  } catch(error) {$('#signal').textContent=String(error);return;}
  if(screen!=='lab'||experiment!=='redact')report.text=text;
  const blob=new Blob([JSON.stringify({reader:state.reader,vessel_register:state.vessels,...report},null,2)],{type:'application/json'});
  const a=el('a');a.href=URL.createObjectURL(blob);a.download='saltglass-log.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};
draw();
try {await init();ready=true;draw();$('#signal').textContent='Anno WebAssembly · '+catalog[state.reader][0];}
catch {$('#signal').textContent='Reader unavailable. Reload to retry.';}
