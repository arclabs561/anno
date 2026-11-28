// Loaded only after an explicit scan from the survey bench.
import {Gliner} from './vendor/gliner.mjs';

const manifest=await fetch(new URL('./model-weights.json',import.meta.url));
if(!manifest.ok)throw new Error('Could not load the survey cartridge record.');
const {survey:cartridge}=await manifest.json();
const {repository,revision,sha256,bytes:size}=cartridge;
const weights='https://huggingface.co/'+repository+'/resolve/'+revision+'/'+cartridge.path;
let model;

async function openReceiver(status){
  status('Opening the 183 MB survey cartridge…');
  let cache;
  try { cache=await caches.open('saltglass-survey-v1'); } catch {}
  const cached=await cache?.match(weights);
  let bytes;
  if(cached){
    bytes=new Uint8Array(await cached.arrayBuffer());
  }else{
    const response=await fetch(weights);
    if(!response.ok)throw new Error('Survey cartridge download failed ('+response.status+').');
    bytes=new Uint8Array(size);
    const stream=response.body.getReader();
    let received=0,shown=-1;
    while(true){
      const {value,done}=await stream.read();
      if(done)break;
      bytes.set(value,received);
      received+=value.length;
      const mb=Math.floor(received/1000000);
      if(mb>=shown+5){status('Receiving cartridge · '+mb+' / 183 MB');shown=mb;}
    }
    if(received!==size)throw new Error('Incomplete survey cartridge. Run the scan again.');
  }
  if(bytes.length!==size){
    await cache?.delete(weights);
    throw new Error('Incomplete cached cartridge. Run the scan again.');
  }
  status('Verifying cartridge checksum…');
  const digest=new Uint8Array(await crypto.subtle.digest('SHA-256',bytes));
  const actual=Array.from(digest,byte=>byte.toString(16).padStart(2,'0')).join('');
  if(actual!==sha256){
    await cache?.delete(weights);
    throw new Error('Cartridge checksum mismatch. Run the scan again to fetch a fresh copy.');
  }
  if(!cached){
    try { await cache?.put(weights,new Response(bytes)); } catch {}
  }
  status('Starting the local neural reader…');
  const receiver=new Gliner({
    tokenizerPath:repository,tokenizerRevision:revision,
    onnxSettings:{modelPath:bytes,executionProvider:'wasm',multiThread:true,maxThreads:1},
    transformersSettings:{useBrowserCache:true}
  });
  await receiver.initialize();
  return receiver;
}

export async function survey(text,labels,threshold,status){
  if(![0.5,0.7,0.85].includes(threshold))throw new Error('Choose a supported score cutoff.');
  if(!text.trim())throw new Error('Enter a passage for the survey.');
  if(Array.from(text).length>1200)throw new Error('Choose a passage of up to 1,200 characters.');
  if(!labels.length || labels.length>8 || labels.some(label=>label.length>40))
    throw new Error('Enter one to eight comma-separated labels, up to 40 characters each.');
  const start=performance.now();
  model ??= await openReceiver(status);
  status('Scanning for '+labels.join(', ')+'…');
  const scan=performance.now();
  const [matches]=await model.inference({texts:[text],entities:labels,threshold});
  const entities=matches.map(({spanText,start,end,label,score})=>({
    text:spanText,label,score,start:Array.from(text.slice(0,start)).length,end:Array.from(text.slice(0,end)).length
  }));
  return {entities,labels,model:repository,revision,weights_sha256:sha256,threshold,
    scan_ms:Math.round(performance.now()-scan),total_ms:Math.round(performance.now()-start)};
}
