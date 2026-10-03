// Presentation logic only; no browser rendering or neural model execution.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element{
 constructor(){this.children=[];this.style={};this.value='';this.checked=false;}
 appendChild(node){this.children.push(node);return node;}
 replaceChildren(){this.children=[];}
 setAttribute(key,value){assert(!String(value).includes('NaN'));}
 addEventListener(){}
}
const points=[];
for(const model of ['flex_quality','nature_cnn','impala_cnn','impoola_cnn'])for(const seed of [1,2])for(const steps of [10,20])
 points.push({model,seed,steps,seconds:steps/10,win_rate:steps===10?.5:.4});
const means=points.filter(p=>p.seed===1).map(p=>({...p,score_low:.1,score_high:.9,time_low:.1,time_high:10}));
const data={purpose:'synthetic chart fixture',status:'complete_descriptive',points,means,missing:[],failures:[]};
const ids=Object.fromEntries(['view','axis','bands','warning','legend','plot','detail'].map(k=>[k,new Element()]));
ids.view.value='mean';ids.axis.value='seconds';
const document={getElementById:k=>ids[k],createElement:()=>new Element(),createElementNS:()=>new Element()};
const html=fs.readFileSync('research/claim_frontier.html','utf8').replace('__DATA__',JSON.stringify(data));
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const context=vm.createContext({document});vm.runInContext(script,context);
for(const view of ['mean','1','2'])for(const axis of ['seconds','steps'])for(const bands of [false,true]){
 ids.view.value=view;ids.axis.value=axis;ids.bands.checked=bands;vm.runInContext('draw()',context);
 assert(ids.plot.children.length>20);
}
ids.view.value='missing';vm.runInContext('draw()',context);assert.equal(ids.plot.children.length,1);
console.log('PASS: 12 full-range chart views and empty data; synthetic fixtures only.');
