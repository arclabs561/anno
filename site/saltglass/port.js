import init, {extract, inspect} from './engine/pkg/saltglass_reader.js';

const $ = selector => document.querySelector(selector);
const log = "From the landing field at Saltglass Reach you can usually see two of the three moons. To find the third, stand beside the mail shed and look through the gap in the observatory stairs. Iona Vale painted a white mark on the step for visitors. She touches it up every spring.\n\nVale captains the Kestrel for the Brackwater Packet Company. Yesterday she brought the company’s new tide tables over from Low Bell, along with one passenger and a crate of dry socks. The tables say the mail road stays dry until noon. The passenger walked in at eleven with his boots full of water.";
const catalog = {
  crf: ['CRF reader', 2, 'A trained sequence model for people, organizations and places. Its weights come from Anno’s English WikiANN model. It reads the words around a name, so unfamiliar ports can have their own entries.'],
  registry: ['Port gazetteer', 0, 'Anno LexiconNER looks up local names in the Saltglass registry. The vessel register adds the Kestrel to its vocabulary.'],
  heuristic: ['Heuristic reader', 1, 'Anno HeuristicNER uses capitalization, nearby words and other cues. Try names and organizations from beyond the Reach.'],
  patterns: ['Pattern reader', 4, 'Anno RegexNER recognizes structured text. Try a date, a sum of money or a (fictional) email address in the reader lab.'],
  vessels: ['Vessel register', 6, 'The harbor supplement lists ships by name. Fit it alongside the port gazetteer to identify the Kestrel in the dispatch.']
};
const readers = ['crf','registry','heuristic','patterns'];
const key = 'anno.saltglass.v1';
let saved;
try { saved = JSON.parse(localStorage.getItem(key)); } catch {}
const state = {
  reader: readers.includes(saved?.reader) ? saved.reader : 'crf',
  vessels: saved?.vessels === true,
  mission: ['accepted','delivered'].includes(saved?.mission) ? saved.mission : 'open',
  notice: saved?.notice==='amended'?'amended':'posted',
  amendment: readers.includes(saved?.amendment)?saved.amendment:null,
  receipt: saved?.receipt===true,
  sample: log,
  away: saved?.away === true
};
let screen = 'log', selected = state.reader, ready = false, labResult = null;
let experiment = 'mentions';
const notices = {
  posted: {number:'41', text:'14 May 2026 · 12:00. The mail road dries at noon. Hold the Kestrel at Low Bell until the carts arrive.'},
  amended: {number:'42', text:'15 May 2026 · 11:00. 🌙 Iona Vale sounded the west berth. She reports the mail road underwater. Send the Kestrel by the Low Bell channel.'}
};
const experiments = {
  mentions:['Mentions', log],
  compare:['Compare readers', '🌙 Iona Vale reached Low Bell. Contact iona@example.org on May 12, 2026.'],
  links:['Linked mentions', 'Iona Vale sailed to Low Bell. Iona Vale returned to Saltglass Reach.'],
  relations:['Relation candidates', 'Iona Vale founded Brackwater Packet Company.'],
  brat:['Standoff manifest (.ann)', '🌙 Iona Vale reached Low Bell.'],
  conll:['Token manifest (BIO)', 'Iona Vale reached Low Bell.'],
  redact:['Redact contacts', 'Reach the fictional harbor clerk at iona@example.org.']
};
function save() { try { localStorage.setItem(key, JSON.stringify({reader:state.reader,vessels:state.vessels,mission:state.mission,notice:state.notice,amendment:state.amendment,receipt:state.receipt,away:state.away})); } catch {} }
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
    const reveal=()=>{
      part.querySelector('[aria-pressed="true"]')?.setAttribute('aria-pressed','false');
      mark.setAttribute('aria-pressed','true');
      detail.textContent=e.text+' · '+type+' · characters ['+e.start+', '+e.end+')';};
    mark.setAttribute('aria-pressed','false');mark.tabIndex=0;mark.setAttribute('role','button');mark.onclick=reveal;
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
let readerModule;
let labWorker = null;
let pending = false;

function cancelAnalysis() {
  if(pending){
    labWorker.terminate();
    labWorker=null;
    pending=false;
  }
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
  } else if('manifest' in report){
    result.append(el('pre',report.manifest || 'No records in this manifest.','manifest'));
  } else if(report.experiment==='redact'){
    result.append(el('p','Pattern-based redaction · review before sharing','small'),el('p',report.redacted));
  } else {
    const links=report.experiment==='links';
    result.append(el('p',links?'Rule-based links among detected mentions':'Heuristic candidates · founded, works for, located in','small'));
    if(!report.groups.length){
      result.append(el('p',links?'No repeated mentions linked.':'No matching relation found.'));
      if(state.reader!=='registry'){
        result.append(action('Fit port gazetteer and rerun',()=>{
          state.reader='registry';labResult=null;save();draw(true);$('#heading').focus();
          $('#signal').textContent='Anno WebAssembly · '+catalog.registry[0];
        }));
      }
    }
    for(const row of report.groups){
      result.append(el('p',links?row.map(m=>m.text+' ['+m.start+', '+m.end+')').join(' ↔ '):[row.head,row.predicate,row.tail].join(' → ')));
    }
  }
}
function draw(rerun=false) {
  cancelAnalysis();
  const content=$('#content');content.replaceChildren();
  $('#port').hidden=state.away;$('#orbit').hidden=!state.away;
  $('#harbor-call').textContent = {open:'Mail road flooded · use west berth', accepted:'West berth · Kestrel awaiting dispatch', delivered:'Channel clear · Kestrel outbound'}[state.mission];
  $('#berth').textContent=state.mission==='delivered'?'Vacant':'Kestrel';
  $('#ship-reader').textContent=catalog[state.reader][0];
  $('#port-fix').textContent=state.away?'On approach':'West berth';
  $('#ship-rail').dataset.outbound=String(state.mission==='delivered');
  $('#orbit-message').textContent=state.mission==='delivered'?'The Kestrel clears the flats with the corrected tide tables.':'A lamp burns above the mail shed. The port is accepting arrivals.';
  $('#heading').textContent=({log:'Saltglass Reach',lab:'Reader lab',board:'Mission BBS'})[screen];
  document.querySelectorAll('[data-screen]').forEach(b=>b.setAttribute('aria-current',String(b.dataset.screen===screen)));
  $('#export').onclick=exportPortLog;
  $('#export').disabled=!ready || (screen==='board' && state.mission!=='accepted' && !state.receipt);
  const formatLabel = {brat:'Download .ann',conll:'Download BIO'};
  $('#export').textContent=screen==='board'?(state.receipt?'Export receipt':'Export dispatch'):(screen==='lab' && formatLabel[experiment])||'Export log';
  if(screen==='log'){
    const rows=ready?read(log):[];
    content.append(annotated(log,rows));
    if(ready){
      const ledger=el('details',undefined,'watch-ledger');
      ledger.append(el('summary','Watch ledger · trace the captain'));
      ledger.ontoggle=()=>{
        if(!ledger.open || ledger.children.length>1)return;
        const chain=JSON.parse(inspect(log,state.reader,state.vessels,'links'));
        const record=el('ol');
        for(const match of log.matchAll(/Iona Vale|\bVale\b|\b[Ss]he\b/g)){
          const start=Array.from(log.slice(0,match.index)).length;
          const end=start+Array.from(match[0]).length;
          const mention=rows.find(e=>e.start===start && e.end===end);
          const linked=chain.find(g=>g.some(m=>m.start===start && m.end===end));
          const status=linked?'linked to '+linked.filter(m=>m.start!==start).map(m=>m.text).join(', '):mention?'detected · '+category(mention.label):'missed';
          record.append(el('li',match[0]+' · '+status));
        }
        ledger.append(el('p','Clerk’s reading, in passage order, checked against the '+catalog[state.reader][0]+'.','small'),record);
      };
      content.append(ledger);
    }
  }
  if(screen==='lab') {
    const input=el('textarea');input.id='sample';input.rows=5;input.maxLength=6000;input.value=state.sample;
    const caption=el('label','Local text · '+catalog[state.reader][0]);caption.htmlFor=input.id;
    const result=el('div');result.setAttribute('aria-live','polite');
    const exporter=$('#export');
    const reset=()=>{
      cancelAnalysis();labResult=null;result.replaceChildren();
      scan.disabled=!ready;exporter.disabled=!ready;
    };
    const run=(download=false)=>{
      if(pending)return;
      const text=input.value;
      if(labResult){
        if(download)saveReport(labResult,text);
        else showAnalysis(result,text,labResult);
        return;
      }
      const fail=message=>{
        cancelAnalysis();result.textContent=message;scan.disabled=false;exporter.disabled=false;
      };
      try {
        const module = labWorker ? undefined : readerModule;
        labWorker ??= new Worker(new URL('./reader-worker.js',import.meta.url),{type:'module'});
        const worker=labWorker;
        pending=true;scan.disabled=true;exporter.disabled=true;
        result.textContent='Reading manifest…';
        worker.onmessage=({data})=>{
          if(worker!==labWorker || !pending)return;
          if(data.error){fail(data.error);return;}
          pending=false;scan.disabled=false;exporter.disabled=false;
          labResult=data.report;
          showAnalysis(result,text,labResult);
          if(download)saveReport(labResult,text);
        };
        worker.onerror=()=>{
          if(worker===labWorker)fail('Reader interrupted. Run Anno to retry.');
        };
        worker.postMessage({module,text,experiment,reader:state.reader,vessels:state.vessels,readers});
      } catch(error){fail(String(error));}
    };
    input.oninput=()=>{state.sample=input.value;reset();};
    const scan=action('Run Anno',()=>run());scan.disabled=!ready;
    exporter.onclick=()=>run(true);
    const selector=el('select');
    selector.id='experiment';
    for(const [value,[name]] of Object.entries(experiments)){
      const option=el('option',name);option.value=value;selector.append(option);
    }
    selector.value=experiment;
    selector.onchange=()=>{experiment=selector.value;$('#export').textContent=formatLabel[experiment]||'Export log';reset();};
    const mode=el('label','Experiment ');mode.append(selector);
    const example=action('Load example',()=>{
      input.value=experiments[experiment][1]; input.oninput(); if(ready)run();
    });
    content.append(mode,caption,input,scan,example,result);
    if(ready && !state.away && (rerun || labResult)) run();
  }
  if(screen==='board') drawBoard(content);
}

function clerkRecord(){
  const text=notices.amended.text;
  const span=word=>{
    const start=Array.from(text.slice(0,text.indexOf(word))).length;
    return {text:word,start,end:start+Array.from(word).length};
  };
  return {source:'harbor clerk',reviewed_reader:state.amendment,
    entity:{...span('Iona Vale'),label:'PER'},coreference:{mention:span('She'),antecedent:span('Iona Vale')}};
}

function drawBoard(content){
  content.append(el('h3','West berth · tide dispatch'));
  if(state.mission==='open'){
    content.append(el('p','Two tide notices are clipped to the mail hook. The Kestrel needs the one that accounts for the flooded road.'),action('Accept dispatch',()=>{state.mission='accepted';save();draw();}));
    return;
  }
  if(state.mission==='delivered'){
    const receipt=el('section',undefined,'receipt');
    receipt.append(el('strong','WEST BERTH  /  DEPARTURE COPY'),el('p',state.receipt?'Notice 42 · 15 May · 11:00 sounding':'Corrected tables received'),el('p','Kestrel → Low Bell'),el('p','Dispatch signed by Iona Vale. Return passage paid.'));
    if(state.receipt)receipt.append(el('small','Clerk verified the captain and linked “She” to her signature.'));
    content.append(receipt);
    if(!state.receipt)content.append(action('Replay with the two tide notices',()=>{
      state.mission='accepted';state.notice='posted';state.amendment=null;save();draw();
    }));
    return;
  }
  const notice=notices[state.notice];
  const slips=el('div',undefined,'slip-tabs');
  for(const [id,n] of Object.entries(notices)){
    const tab=action('Notice '+n.number,()=>{state.notice=id;save();draw();document.querySelector('.slip-tabs [aria-pressed=true]').focus();});
    tab.setAttribute('aria-pressed',String(state.notice===id));slips.append(tab);
  }
  const paper=el('section',undefined,'notice-paper');
  const spans=ready?read(notice.text):[];
  paper.append(el('strong','SALTGLASS  /  SOUNDINGS OFFICE'),annotated(notice.text,spans));
  content.append(slips,paper);
  if(state.notice==='amended' && ready){
    const original=spans.find(e=>e.text==='Iona Vale');
    content.append(el('p',catalog[state.reader][0]+' reads the signature as '+(original?category(original.label):'unmarked')+'.','small'));
    if(state.amendment)content.append(el('p','Clerk’s pencil · Iona Vale = person · She → Iona Vale','pencil'),action('Erase amendment',()=>{state.amendment=null;save();draw();}));
    else content.append(action('Mark Iona Vale as person and link “She”',()=>{state.amendment=state.reader;save();draw();}));
  }
  const feedback=el('p');feedback.setAttribute('role','status');
  const send=action('Deliver notice '+notice.number,()=>{
    if(state.notice==='posted'){feedback.textContent='Held at the mail hook. This is yesterday’s prediction. Find the observation from the flooded berth.';return;}
    if(!state.amendment){feedback.textContent='The clerk needs a named witness. Check the signature and the following pronoun.';return;}
    if(!spans.some(e=>category(e.label)==='vessel')){feedback.textContent='Ship unverified. Fit the vessel register in the Outfitter to read the addressee.';return;}
    state.mission='delivered';state.receipt=true;save();draw();$('#heading').focus();
  });
  send.disabled=!ready;content.append(send,feedback);
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
  const example=screen==='board' && state.mission==='accepted'?notices[state.notice].text:'Vale captains the Kestrel for the Brackwater Packet Company.';
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
function exportPortLog(){
  const text=screen==='board'?notices[state.notice].text:log;
  const report={entities:read(text)};
  if(screen==='board'){
    report.notice=notices[state.notice].number;
    if(state.notice==='amended' && state.amendment)report.amendment=clerkRecord();
    if(state.receipt)report.receipt={signed_by:'Iona Vale',vessel:'Kestrel',destination:'Low Bell'};
  }
  saveReport(report,text);
}
function saveReport(report,text){
  const source=report.experiment==='redact'?{}:{text};
  const manifest='manifest' in report;
  const blob=new Blob([manifest?report.manifest:JSON.stringify({reader:state.reader,vessel_register:state.vessels,...report,...source},null,2)],{type:manifest?'text/plain':'application/json'});
  const a=el('a');a.href=URL.createObjectURL(blob);a.download=manifest?(experiment==='brat'?'saltglass.ann':'saltglass.bio'):'saltglass-log.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};
draw();
try {
  const response=await fetch(new URL('./engine/pkg/saltglass_reader_bg.wasm',import.meta.url));
  if(!response.ok)throw new Error('Reader download failed');
  readerModule=WebAssembly.compileStreaming && response.headers.get('Content-Type')?.split(';')[0]==='application/wasm'
    ? await WebAssembly.compileStreaming(response)
    : await WebAssembly.compile(await response.arrayBuffer());
  await init({module_or_path:readerModule});ready=true;draw();$('#signal').textContent='Anno WebAssembly · '+catalog[state.reader][0];}
catch {$('#signal').textContent='Reader unavailable. Reload to retry.';}
