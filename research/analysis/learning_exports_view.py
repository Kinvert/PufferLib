"""Self-contained exploratory view of exported game means, not paper figures."""
import json


def write(path, value):
    data = json.dumps(value).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>5090 checkpoint frontiers — exported means</title>
<style>body{font:16px system-ui;background:#101821;color:#e6edf3;max-width:1100px;margin:25px auto;padding:15px}select{font:inherit}#models{display:flex;flex-wrap:wrap;gap:12px;margin:15px 0}label{cursor:pointer}svg{background:#162331;width:100%;height:auto}p{line-height:1.5}a{color:#72b9ff}</style>
<h1>5090: complete checkpoint curves</h1>
<p>Two paired discovery seeds, all drawings averaged within each game. Every scheduled checkpoint and every model remains in the data. These are descriptive exported means; raw episode receipts have not been independently audited here.</p>
<select id="game"></select> <label><input type="checkbox" id="log"> Log training time</label>
<div id="models"></div><svg id="plot" viewBox="0 0 1000 570" role="img" aria-label="Training time and score curves"></svg>
<p id="metric"></p><p>Bold marker outlines identify the lower-endpoint frontier computed across <b>all</b> models and checkpoints. Filtering does not recompute frontiers or axis ranges. Hover a marker for its model, decisions, cost and bounds. Vertical lines show Pong censoring bounds, <b>not confidence intervals</b>.</p>
<p>The aggregate option contains only final-budget points. Game plots contain all four checkpoints; declines are preserved. Native checkpoint cost excludes evaluation and is charged once per training job. No statistical dominance or SOTA is certified.</p>
<script>
const data=__DATA__;
const palette=['#5be0b6','#ffd166','#80b6ff','#ff7a90','#bc9cff','#80d8e5','#f6a85f','#cce17a','#d0a7ec','#8bc39b','#dea4a4','#799dc7','#d7d4ab','#d97fcc'];
const names=['quality-reference','nature-cnn',...data.models.filter(x=>!['quality-reference','nature-cnn'].includes(x))];
const selected=new Set(names), colors=Object.fromEntries(names.map((n,i)=>[n,palette[i]]));
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const select=document.getElementById('game');
for(const g of ['connect4cnn','pongcnn','flappycnn','breakoutcnn','snakecnn','mazecnn','aggregate']){const o=document.createElement('option');o.value=g;o.textContent=g;select.append(o);}
for(const n of names){const l=document.createElement('label'),c=document.createElement('input');c.type='checkbox';c.checked=true;c.onchange=()=>{c.checked?selected.add(n):selected.delete(n);draw();};l.append(c,document.createTextNode(n));l.style.color=colors[n];document.getElementById('models').append(l);}
function draw(){
 const aggregate=select.value==='aggregate', all=aggregate?data.aggregate_final:data.by_game[select.value];
 const logarithmic=document.getElementById('log').checked, transform=x=>logarithmic?Math.log10(x):x;
 const xmin=logarithmic?Math.min(...all.map(p=>transform(p.seconds))):0,xmax=Math.max(...all.map(p=>transform(p.seconds)));
 const ymax=Math.max(.001,...all.map(p=>p.score_upper))*1.05;
 const x=s=>80+860*(transform(s)-xmin)/(xmax-xmin),y=s=>490-420*s/ymax;
 let svg='';
 for(let i=0;i<=5;i++){const xx=80+860*i/5,yy=490-420*i/5,tx=xmin+(xmax-xmin)*i/5;svg+=`<path d="M80 ${yy}H940 M${xx} 70V490" stroke="#354453"/><text x="68" y="${yy+5}" text-anchor="end" fill="#e6edf3">${(ymax*i/5).toFixed(3)}</text><text x="${xx}" y="515" text-anchor="middle" fill="#e6edf3">${(logarithmic?10**tx:tx).toFixed(1)}</text>`;}
 for(const name of names){if(!selected.has(name))continue;const points=all.filter(p=>p.candidate===name).sort((a,b)=>(a.decisions||0)-(b.decisions||0));
  if(!aggregate)svg+=`<polyline points="${points.map(p=>`${x(p.seconds)},${y(p.score_lower)}`).join(' ')}" fill="none" stroke="${colors[name]}" stroke-width="1.7"/>`;
  for(const p of points){const frontier=aggregate?p.frontier:p.frontier_lower_endpoint;const title=`${name}; ${p.decisions||'final budget'} decisions; ${p.seconds.toFixed(3)}s; score ${p.score_lower.toFixed(5)} to ${p.score_upper.toFixed(5)}`;
   svg+=`<path d="M${x(p.seconds)} ${y(p.score_lower)}V${y(p.score_upper)}" stroke="${colors[name]}" stroke-width="2"/><circle cx="${x(p.seconds)}" cy="${y(p.score_lower)}" r="${frontier?6:4}" fill="${colors[name]}" stroke="${frontier?'#ffffff':colors[name]}" stroke-width="${frontier?2:1}"><title>${esc(title)}</title></circle>`;
  }
 }
 svg+='<text x="510" y="553" text-anchor="middle" fill="#e6edf3">'+(aggregate?'Total training seconds, both seeds/all six games':'Mean checkpoint training seconds per seed')+'</text>';
 document.getElementById('plot').innerHTML=svg;
 document.getElementById('metric').textContent=aggregate?'Fixed-anchor equal-game final score; clipping performed before averaging.':all[0].metric;
}
select.onchange=draw;document.getElementById('log').onchange=draw;draw();
</script></html>'''
    path.write_text(page.replace("__DATA__", data))
