const fs=require('fs');
const WebSocket=require('ws');
const ROOT=process.argv[2];if(!ROOT)throw Error('Usage: node check_browser.cjs GALLERY_DIR [URL] [DEBUG_PORT]');
const URL=process.argv[3]||'http://127.0.0.1:8875/index.html';
async function main(){
  const pages=await(await fetch('http://127.0.0.1:'+(process.argv[4]||9234)+'/json')).json();
  const page=pages.find(p=>p.type==='page');const ws=new WebSocket(page.webSocketDebuggerUrl);
  await new Promise(resolve=>ws.once('open',resolve));let id=0;const pending=new Map(),exceptions=[];
  ws.on('message',raw=>{const v=JSON.parse(raw);if(v.id){const task=pending.get(v.id);pending.delete(v.id);v.error?task.reject(Error(JSON.stringify(v.error))):task.resolve(v.result);}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params);});
  function command(method,params={}){return new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});}
  async function evaluate(expression){const r=await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
  await command('Page.enable');await command('Runtime.enable');
  await command('Emulation.setDeviceMetricsOverride',{width:1450,height:1200,deviceScaleFactor:1,mobile:false});
  await command('Page.navigate',{url:URL});
  for(let n=0;n<100;n++){if(await evaluate("typeof currentPixels!=='undefined' && currentPixels!==null"))break;await new Promise(r=>setTimeout(r,50));}
  const report=await evaluate(`(async()=>{
    const checks=[];
    const expected=3*Object.values(DATA.games).reduce((n,g)=>n+g.names.length*(g.modes||['rgb','legacy']).length,0);
    if(DATA.frames.length!==expected)throw Error('Missing gallery frames');
    for(let gi=0;gi<keys.length;gi++){
      $('games').children[gi].click();await render();
      for(const input of (DATA.games[game].modes||['rgb','legacy'])){
        $('mode').value=input;$('mode').dispatchEvent(new Event('change'));await render();
        $('all').click();await render();
        const cards=[...document.querySelectorAll('#contact .card')];
        if(cards.length!==DATA.games[game].names.length)throw Error('Missing variant cards');
        await Promise.all(cards.map(c=>c.querySelector('img').decode()));
        if(cards.some(c=>!c.querySelector('img').naturalWidth))throw Error('Missing frame image');
        for(let i=0;i<cards.length;i++){
          cards[i].click();await render();
          if($('image').width!==frame().shape_hwc[1]||$('image').height!==frame().shape_hwc[0])throw Error('Wrong canvas dimensions');
          $('all').click();await render();
        }
        checks.push({game,input,variants:cards.length,all_images_loaded:true});
      }
    }
    game='connect4cnn';rep=5;mode='rgb';moment=1;view='single';await render();
    $('zoom').value='2';$('zoom').dispatchEvent(new Event('change'));
    if(parseFloat($('image').style.width)!==1344)throw Error('Zoom failed');
    $('zoom').value='fit';zoom();
    const rect=$('image').getBoundingClientRect();
    $('image').dispatchEvent(new MouseEvent('mousemove',{clientX:rect.left+rect.width/2,clientY:rect.top+rect.height/2}));
    if(!$('pixel').textContent.includes('input RGB=['))throw Error('Pixel inspection failed');
    $('moment').value='2';$('moment').dispatchEvent(new Event('change'));await render();
    if(frame().state.pieces!==14)throw Error('Moment selector failed');
    document.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));await render();
    if(rep!==6)throw Error('Keyboard navigation failed');
    $('compare').click();await render();
    const comparison=[...document.querySelectorAll('#comparison canvas')];
    if(comparison.length!==2||comparison[0].width!==672||comparison[1].width!==44)throw Error('Comparison failed');
    if($('error').textContent)throw Error($('error').textContent);
    // Read every PNG through a browser, including all three moments.
    for(const f of DATA.frames){const image=new Image();image.src=f.image;await image.decode();if(image.naturalWidth!==f.shape_hwc[1]||image.naturalHeight!==f.shape_hwc[0])throw Error('Wrong PNG shape');}
    game=keys.includes('foragecnn')?'foragecnn':'connect4cnn';rep=0;mode='rgb';moment=1;view='single';await render();
    return {status:'ok',browser:'Chrome headless, software rendering',frames:DATA.frames.length,checks,pixel_inspector:true,zoom:true,moment_selection:true,keyboard:true,comparison:true};
  })()`);
  if(exceptions.length)throw Error(JSON.stringify(exceptions));
  const shot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  fs.writeFileSync(ROOT+'/gallery-preview.png',Buffer.from(shot.data,'base64'));
  fs.writeFileSync(ROOT+'/browser-check.json',JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report,null,2));
  await command('Browser.close');ws.close();
}
main().catch(e=>{console.error(e);process.exit(1)});
