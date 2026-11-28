import {initSync, extract, inspect} from './engine/pkg/saltglass_reader.js';

function analyze({text,reader,vessels,experiment,readers}) {
  const read = (text,id=reader) => JSON.parse(extract(text,id,vessels));
  if(experiment==='compare') {
    return {experiment,readers:Object.fromEntries(readers.map(id=>[id,read(text,id)]))};
  }
  if(experiment==='mentions') return {experiment,entities:read(text)};
  const value=inspect(text,reader,vessels,experiment);
  if(experiment==='brat'||experiment==='conll') return {experiment,manifest:value};
  return experiment==='redact'?{experiment,redacted:value}:{experiment,groups:JSON.parse(value)};
}

self.onmessage = async ({data}) => {
  try {
    if(data.module)initSync({module:data.module});
    if(data.experiment==='neural'){
      const {survey}=await import('./neural-reader.js');
      const report=await survey(data.text,data.labels,data.threshold,status=>self.postMessage({status}));
      const baseline=JSON.parse(extract(data.text,data.reader,data.vessels));
      self.postMessage({report:{...report,experiment:'neural',baseline,baseline_reader:data.reader}});
    }else self.postMessage({report:analyze(data)});
  } catch (error) {
    self.postMessage({error:String(error)});
  }
};
