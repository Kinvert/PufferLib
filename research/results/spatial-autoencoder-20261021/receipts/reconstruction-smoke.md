# Spatial autoencoder reconstruction pilot

Full RGB -> spatial feature grid -> decoder. No classification head was trained.
Object/marking errors are instance means, split by class and skin. No RL/Pareto claim.

|Stride|Channels|Latent HWC|Loss|Updates|Train process seconds|Images/s|
|---|---|---|---|---|---|---|
|8|8|[60, 80, 8]|objects|4|0.516|7.76|

## Stride 8, channels 8, objects

|Split|Skin|Object|Instances|Object MSE|Mark MSE|Background MSE|
|---|---|---|---|---|---|---|
|train|0|ground|0|N/A|N/A|0.02487386|
|train|0|apple|6|0.19151965|0.34843087|0.02529446|
|train|0|pear|6|0.38116218|N/A|0.025472867|
|train|0|grapes|2|0.15963926|N/A|0.025586069|
|train|0|leaf|6|0.23158009|N/A|0.025017853|
|train|0|flower|6|0.19622868|N/A|0.025203488|
|train|0|beetle|6|0.17969343|N/A|0.025196295|
|train|0|striped mushroom|6|0.23315075|0.3480593|0.024855809|
|train|0|spotted apple|6|0.1914411|0.34828924|0.025294342|
|train|0|wasp|6|0.26479713|N/A|0.025050383|
|train|0|spotted toadstool|6|0.23312699|0.34801041|0.024846551|
|train|0|tree|148|0.12585254|N/A|0.025056666|
|train|0|explorer|2|0.2049247|N/A|0.025746508|
|validation|0|ground|0|N/A|N/A|0.02487411|
|validation|0|apple|6|0.19148088|0.34839652|0.025285932|
|validation|0|pear|6|0.38117943|N/A|0.025474128|
|validation|0|grapes|2|0.15963896|N/A|0.025584783|
|validation|0|leaf|6|0.2315806|N/A|0.025023576|
|validation|0|flower|6|0.19623989|N/A|0.025195885|
|validation|0|beetle|6|0.17972306|N/A|0.025198561|
|validation|0|striped mushroom|6|0.23317251|0.34807882|0.024861835|
|validation|0|spotted apple|6|0.19151148|0.34842039|0.025300865|
|validation|0|wasp|6|0.26478231|N/A|0.025056145|
|validation|0|spotted toadstool|6|0.23315039|0.34804958|0.024849477|
|validation|0|tree|148|0.12585388|N/A|0.025055788|
|validation|0|explorer|2|0.20491287|N/A|0.025726131|
|test|0|ground|0|N/A|N/A|0.024874155|
|test|0|apple|6|0.19148044|0.34839216|0.025287865|
|test|0|pear|6|0.38111828|N/A|0.025470291|
|test|0|grapes|2|0.1596499|N/A|0.025594066|
|test|0|leaf|6|0.23161678|N/A|0.025020504|
|test|0|flower|6|0.19622982|N/A|0.025197078|
|test|0|beetle|6|0.17968895|N/A|0.025195583|
|test|0|striped mushroom|6|0.23314946|0.34805769|0.024854475|
|test|0|spotted apple|6|0.19148301|0.34838058|0.025288533|
|test|0|wasp|6|0.26478807|N/A|0.025046296|
|test|0|spotted toadstool|6|0.23314385|0.34801749|0.024847197|
|test|0|tree|148|0.12585241|N/A|0.025056254|
|test|0|explorer|2|0.20493115|N/A|0.025726021|
|8|8|[60, 80, 8]|plain|4|0.515|7.76|

## Stride 8, channels 8, plain

|Split|Skin|Object|Instances|Object MSE|Mark MSE|Background MSE|
|---|---|---|---|---|---|---|
|train|0|ground|0|N/A|N/A|0.024865062|
|train|0|apple|6|0.1917437|0.34877609|0.025306663|
|train|0|pear|6|0.38170662|N/A|0.025488614|
|train|0|grapes|2|0.15972432|N/A|0.02562441|
|train|0|leaf|6|0.23186418|N/A|0.025028572|
|train|0|flower|6|0.1963299|N/A|0.025202488|
|train|0|beetle|6|0.17970866|N/A|0.025216649|
|train|0|striped mushroom|6|0.23342636|0.34859684|0.024875285|
|train|0|spotted apple|6|0.19171545|0.34870934|0.02530811|
|train|0|wasp|6|0.2649967|N/A|0.025071851|
|train|0|spotted toadstool|6|0.23342418|0.3485547|0.024873166|
|train|0|tree|148|0.12597197|N/A|0.025057419|
|train|0|explorer|2|0.20511445|N/A|0.025769295|
|validation|0|ground|0|N/A|N/A|0.024865349|
|validation|0|apple|6|0.19173061|0.34876342|0.025303662|
|validation|0|pear|6|0.3817134|N/A|0.025489879|
|validation|0|grapes|2|0.15972904|N/A|0.0256185|
|validation|0|leaf|6|0.23187039|N/A|0.025024801|
|validation|0|flower|6|0.19631961|N/A|0.025194848|
|validation|0|beetle|6|0.17972912|N/A|0.025219975|
|validation|0|striped mushroom|6|0.23344317|0.34860319|0.024871026|
|validation|0|spotted apple|6|0.19172688|0.34872957|0.025310322|
|validation|0|wasp|6|0.26499598|N/A|0.025072211|
|validation|0|spotted toadstool|6|0.233432|0.34856201|0.024876389|
|validation|0|tree|148|0.12597169|N/A|0.025058099|
|validation|0|explorer|2|0.205106|N/A|0.025760928|
|test|0|ground|0|N/A|N/A|0.024865341|
|test|0|apple|6|0.19173175|0.34876666|0.025303155|
|test|0|pear|6|0.38169558|N/A|0.025486492|
|test|0|grapes|2|0.15971884|N/A|0.025612295|
|test|0|leaf|6|0.23187213|N/A|0.025025707|
|test|0|flower|6|0.19631629|N/A|0.025200719|
|test|0|beetle|6|0.17970544|N/A|0.025219824|
|test|0|striped mushroom|6|0.23342467|0.34859278|0.02487488|
|test|0|spotted apple|6|0.19171414|0.34873243|0.025309744|
|test|0|wasp|6|0.26499351|N/A|0.025068672|
|test|0|spotted toadstool|6|0.23343867|0.34853309|0.024873289|
|test|0|tree|148|0.12597136|N/A|0.025058789|
|test|0|explorer|2|0.20510975|N/A|0.025754284|
|16|8|[30, 40, 8]|objects|4|0.516|7.76|

## Stride 16, channels 8, objects

|Split|Skin|Object|Instances|Object MSE|Mark MSE|Background MSE|
|---|---|---|---|---|---|---|
|train|0|ground|0|N/A|N/A|0.024876113|
|train|0|apple|6|0.19146709|0.3485228|0.025280953|
|train|0|pear|6|0.38129479|N/A|0.025465233|
|train|0|grapes|2|0.15965774|N/A|0.025585868|
|train|0|leaf|6|0.23184493|N/A|0.025045434|
|train|0|flower|6|0.19618032|N/A|0.025196141|
|train|0|beetle|6|0.17976587|N/A|0.025235496|
|train|0|striped mushroom|6|0.23322558|0.34812413|0.024841372|
|train|0|spotted apple|6|0.19147101|0.34848641|0.02526904|
|train|0|wasp|6|0.26483334|N/A|0.025062768|
|train|0|spotted toadstool|6|0.2332176|0.34814952|0.02484669|
|train|0|tree|148|0.12587932|N/A|0.025074771|
|train|0|explorer|2|0.20491022|N/A|0.025741131|
|validation|0|ground|0|N/A|N/A|0.024875977|
|validation|0|apple|6|0.19144972|0.34851922|0.025267909|
|validation|0|pear|6|0.38129007|N/A|0.025462562|
|validation|0|grapes|2|0.15963946|N/A|0.025560178|
|validation|0|leaf|6|0.23185136|N/A|0.025048607|
|validation|0|flower|6|0.19617144|N/A|0.025192331|
|validation|0|beetle|6|0.17977145|N/A|0.025236045|
|validation|0|striped mushroom|6|0.23322458|0.34810468|0.024851884|
|validation|0|spotted apple|6|0.19146788|0.34845424|0.025282804|
|validation|0|wasp|6|0.2648209|N/A|0.02505897|
|validation|0|spotted toadstool|6|0.23322301|0.34815042|0.024847316|
|validation|0|tree|148|0.12588088|N/A|0.02507455|
|validation|0|explorer|2|0.20491361|N/A|0.025732892|
|test|0|ground|0|N/A|N/A|0.024875856|
|test|0|apple|6|0.19146004|0.34852286|0.025270758|
|test|0|pear|6|0.38125323|N/A|0.025453048|
|test|0|grapes|2|0.15965273|N/A|0.025570122|
|test|0|leaf|6|0.23187581|N/A|0.025046365|
|test|0|flower|6|0.19616965|N/A|0.025190798|
|test|0|beetle|6|0.17976388|N/A|0.025232407|
|test|0|striped mushroom|6|0.23322337|0.34812133|0.024843916|
|test|0|spotted apple|6|0.19148815|0.34850037|0.025270447|
|test|0|wasp|6|0.26482779|N/A|0.02506014|
|test|0|spotted toadstool|6|0.23323268|0.34814133|0.024863368|
|test|0|tree|148|0.12587934|N/A|0.025074548|
|test|0|explorer|2|0.20492776|N/A|0.025732011|
|16|8|[30, 40, 8]|plain|4|0.515|7.76|

## Stride 16, channels 8, plain

|Split|Skin|Object|Instances|Object MSE|Mark MSE|Background MSE|
|---|---|---|---|---|---|---|
|train|0|ground|0|N/A|N/A|0.024854241|
|train|0|apple|6|0.19170465|0.34879579|0.025292511|
|train|0|pear|6|0.3816885|N/A|0.025460882|
|train|0|grapes|2|0.15971014|N/A|0.025586799|
|train|0|leaf|6|0.23194715|N/A|0.025024646|
|train|0|flower|6|0.19623179|N/A|0.02515599|
|train|0|beetle|6|0.17973948|N/A|0.025214163|
|train|0|striped mushroom|6|0.23345022|0.34840304|0.024856087|
|train|0|spotted apple|6|0.19170317|0.3487885|0.025297263|
|train|0|wasp|6|0.26499946|N/A|0.025055538|
|train|0|spotted toadstool|6|0.233443|0.34843807|0.024858823|
|train|0|tree|148|0.12595065|N/A|0.025036922|
|train|0|explorer|2|0.20509626|N/A|0.025749002|
|validation|0|ground|0|N/A|N/A|0.024854308|
|validation|0|apple|6|0.19169842|0.34879302|0.025291959|
|validation|0|pear|6|0.38168156|N/A|0.025458337|
|validation|0|grapes|2|0.15969738|N/A|0.025592579|
|validation|0|leaf|6|0.23195615|N/A|0.025018376|
|validation|0|flower|6|0.19623777|N/A|0.025166326|
|validation|0|beetle|6|0.17974954|N/A|0.025214407|
|validation|0|striped mushroom|6|0.23344465|0.34839846|0.024854091|
|validation|0|spotted apple|6|0.19170048|0.34876949|0.025296589|
|validation|0|wasp|6|0.26499675|N/A|0.02505448|
|validation|0|spotted toadstool|6|0.23344528|0.34843991|0.024858873|
|validation|0|tree|148|0.12595106|N/A|0.025036894|
|validation|0|explorer|2|0.20509488|N/A|0.025744155|
|test|0|ground|0|N/A|N/A|0.024854818|
|test|0|apple|6|0.1916991|0.34879557|0.02529271|
|test|0|pear|6|0.38167309|N/A|0.0254542|
|test|0|grapes|2|0.15969446|N/A|0.025590166|
|test|0|leaf|6|0.23197905|N/A|0.025017513|
|test|0|flower|6|0.19622788|N/A|0.025161699|
|test|0|beetle|6|0.17973934|N/A|0.025215011|
|test|0|striped mushroom|6|0.23344377|0.34839681|0.02485341|
|test|0|spotted apple|6|0.19169393|0.34876768|0.025293375|
|test|0|wasp|6|0.26499843|N/A|0.025058097|
|test|0|spotted toadstool|6|0.23344904|0.34843099|0.024868409|
|test|0|tree|148|0.12594905|N/A|0.025036375|
|test|0|explorer|2|0.20510769|N/A|0.025740268|
|16|16|[30, 40, 16]|objects|4|0.466|8.59|

## Stride 16, channels 16, objects

|Split|Skin|Object|Instances|Object MSE|Mark MSE|Background MSE|
|---|---|---|---|---|---|---|
|train|0|ground|0|N/A|N/A|0.024805408|
|train|0|apple|6|0.19135374|0.34799822|0.025255343|
|train|0|pear|6|0.38136164|N/A|0.025457325|
|train|0|grapes|2|0.15967047|N/A|0.025585202|
|train|0|leaf|6|0.23173496|N/A|0.024975177|
|train|0|flower|6|0.19609303|N/A|0.025127635|
|train|0|beetle|6|0.17958456|N/A|0.025121868|
|train|0|striped mushroom|6|0.23323121|0.3480339|0.024823899|
|train|0|spotted apple|6|0.19138226|0.3479759|0.025270881|
|train|0|wasp|6|0.26501189|N/A|0.025040037|
|train|0|spotted toadstool|6|0.23323081|0.34805093|0.024826511|
|train|0|tree|148|0.1257904|N/A|0.025000899|
|train|0|explorer|2|0.20493382|N/A|0.025687397|
|validation|0|ground|0|N/A|N/A|0.02480602|
|validation|0|apple|6|0.19139961|0.34805864|0.025256801|
|validation|0|pear|6|0.38137946|N/A|0.025458009|
|validation|0|grapes|2|0.15967961|N/A|0.025582377|
|validation|0|leaf|6|0.23174412|N/A|0.024972631|
|validation|0|flower|6|0.19605997|N/A|0.025126461|
|validation|0|beetle|6|0.17957957|N/A|0.025115644|
|validation|0|striped mushroom|6|0.23322341|0.34804312|0.024820129|
|validation|0|spotted apple|6|0.19132434|0.3478728|0.025258388|
|validation|0|wasp|6|0.26503311|N/A|0.025044264|
|validation|0|spotted toadstool|6|0.23324412|0.3480721|0.024829709|
|validation|0|tree|148|0.12578819|N/A|0.024999976|
|validation|0|explorer|2|0.20494834|N/A|0.025691206|
|test|0|ground|0|N/A|N/A|0.024806044|
|test|0|apple|6|0.19139768|0.34805783|0.025264592|
|test|0|pear|6|0.38141603|N/A|0.025455461|
|test|0|grapes|2|0.15967047|N/A|0.025589166|
|test|0|leaf|6|0.23172909|N/A|0.02497154|
|test|0|flower|6|0.1960693|N/A|0.025124154|
|test|0|beetle|6|0.17959552|N/A|0.025122578|
|test|0|striped mushroom|6|0.23323824|0.34805012|0.024825117|
|test|0|spotted apple|6|0.19132446|0.34785039|0.025260622|
|test|0|wasp|6|0.26502643|N/A|0.025042856|
|test|0|spotted toadstool|6|0.23321488|0.34806561|0.02482067|
|test|0|tree|148|0.12578768|N/A|0.024999899|
|test|0|explorer|2|0.2049469|N/A|0.025688203|
|16|16|[30, 40, 16]|plain|4|0.566|7.07|

## Stride 16, channels 16, plain

|Split|Skin|Object|Instances|Object MSE|Mark MSE|Background MSE|
|---|---|---|---|---|---|---|
|train|0|ground|0|N/A|N/A|0.024816832|
|train|0|apple|6|0.19149083|0.348256|0.025262727|
|train|0|pear|6|0.38152163|N/A|0.025465663|
|train|0|grapes|2|0.15969954|N/A|0.025596811|
|train|0|leaf|6|0.23178703|N/A|0.024989231|
|train|0|flower|6|0.19612133|N/A|0.025142909|
|train|0|beetle|6|0.17962295|N/A|0.025140139|
|train|0|striped mushroom|6|0.23335181|0.34824883|0.024845885|
|train|0|spotted apple|6|0.19153374|0.34826715|0.025284225|
|train|0|wasp|6|0.26509944|N/A|0.025062283|
|train|0|spotted toadstool|6|0.23335072|0.3482585|0.024847892|
|train|0|tree|148|0.12583264|N/A|0.025013031|
|train|0|explorer|2|0.20498213|N/A|0.025704475|
|validation|0|ground|0|N/A|N/A|0.024817461|
|validation|0|apple|6|0.19152953|0.34829877|0.025267355|
|validation|0|pear|6|0.38153287|N/A|0.02546565|
|validation|0|grapes|2|0.15970926|N/A|0.025594641|
|validation|0|leaf|6|0.23179736|N/A|0.024987092|
|validation|0|flower|6|0.19609106|N/A|0.025141593|
|validation|0|beetle|6|0.17961991|N/A|0.025133816|
|validation|0|striped mushroom|6|0.23334736|0.34826245|0.024842527|
|validation|0|spotted apple|6|0.19148362|0.34818338|0.025269937|
|validation|0|wasp|6|0.26511205|N/A|0.025065314|
|validation|0|spotted toadstool|6|0.23335633|0.34825693|0.024850183|
|validation|0|tree|148|0.12583042|N/A|0.025011923|
|validation|0|explorer|2|0.20499701|N/A|0.025707976|
|test|0|ground|0|N/A|N/A|0.024817436|
|test|0|apple|6|0.19152546|0.34829349|0.025272877|
|test|0|pear|6|0.38156142|N/A|0.025463378|
|test|0|grapes|2|0.15970177|N/A|0.025602307|
|test|0|leaf|6|0.23178701|N/A|0.024986162|
|test|0|flower|6|0.19609956|N/A|0.025139279|
|test|0|beetle|6|0.17963152|N/A|0.025140826|
|test|0|striped mushroom|6|0.23335831|0.34826375|0.02484679|
|test|0|spotted apple|6|0.1914919|0.34817851|0.025275106|
|test|0|wasp|6|0.26511373|N/A|0.025066348|
|test|0|spotted toadstool|6|0.23332653|0.3482499|0.024843161|
|test|0|tree|148|0.12583024|N/A|0.02501201|
|test|0|explorer|2|0.20499402|N/A|0.025707257|
|32|16|[15, 20, 16]|objects|4|0.516|7.76|

## Stride 32, channels 16, objects

|Split|Skin|Object|Instances|Object MSE|Mark MSE|Background MSE|
|---|---|---|---|---|---|---|
|train|0|ground|0|N/A|N/A|0.024812088|
|train|0|apple|6|0.19154828|0.34842499|0.025284676|
|train|0|pear|6|0.38145042|N/A|0.025466092|
|train|0|grapes|2|0.15970158|N/A|0.02558138|
|train|0|leaf|6|0.23189143|N/A|0.025015242|
|train|0|flower|6|0.19614106|N/A|0.025142289|
|train|0|beetle|6|0.17959553|N/A|0.025135664|
|train|0|striped mushroom|6|0.23290418|0.34759043|0.024747405|
|train|0|spotted apple|6|0.19146027|0.34827859|0.025259515|
|train|0|wasp|6|0.26485632|N/A|0.025027063|
|train|0|spotted toadstool|6|0.23292011|0.34761678|0.024764023|
|train|0|tree|148|0.12583386|N/A|0.025026153|
|train|0|explorer|2|0.20508596|N/A|0.025743345|
|validation|0|ground|0|N/A|N/A|0.02481215|
|validation|0|apple|6|0.19155033|0.34843034|0.025279842|
|validation|0|pear|6|0.38144861|N/A|0.025463865|
|validation|0|grapes|2|0.15971006|N/A|0.025593773|
|validation|0|leaf|6|0.23188978|N/A|0.025016516|
|validation|0|flower|6|0.19612882|N/A|0.025137624|
|validation|0|beetle|6|0.17959803|N/A|0.025141249|
|validation|0|striped mushroom|6|0.23291528|0.34760842|0.024756809|
|validation|0|spotted apple|6|0.19144889|0.34825808|0.025252567|
|validation|0|wasp|6|0.26485686|N/A|0.02502444|
|validation|0|spotted toadstool|6|0.23292362|0.34762176|0.024764187|
|validation|0|tree|148|0.12583379|N/A|0.025025878|
|validation|0|explorer|2|0.20508506|N/A|0.025741903|
|test|0|ground|0|N/A|N/A|0.024812018|
|test|0|apple|6|0.19155351|0.34843471|0.025284558|
|test|0|pear|6|0.38146204|N/A|0.025467993|
|test|0|grapes|2|0.15971261|N/A|0.02559924|
|test|0|leaf|6|0.23189282|N/A|0.025014006|
|test|0|flower|6|0.19612947|N/A|0.025139518|
|test|0|beetle|6|0.17959538|N/A|0.02513672|
|test|0|striped mushroom|6|0.23290757|0.34759487|0.024755578|
|test|0|spotted apple|6|0.19146152|0.34828234|0.025261868|
|test|0|wasp|6|0.26485649|N/A|0.025022166|
|test|0|spotted toadstool|6|0.23292749|0.34762828|0.024763364|
|test|0|tree|148|0.12583325|N/A|0.025026057|
|test|0|explorer|2|0.20508708|N/A|0.025748779|
|32|16|[15, 20, 16]|plain|4|0.515|7.76|

## Stride 32, channels 16, plain

|Split|Skin|Object|Instances|Object MSE|Mark MSE|Background MSE|
|---|---|---|---|---|---|---|
|train|0|ground|0|N/A|N/A|0.024812577|
|train|0|apple|6|0.19167911|0.34863825|0.02530996|
|train|0|pear|6|0.38154964|N/A|0.025473485|
|train|0|grapes|2|0.15972994|N/A|0.02559871|
|train|0|leaf|6|0.23198415|N/A|0.025030562|
|train|0|flower|6|0.19619814|N/A|0.025151187|
|train|0|beetle|6|0.17956855|N/A|0.025142652|
|train|0|striped mushroom|6|0.23301026|0.34778201|0.024762869|
|train|0|spotted apple|6|0.19157805|0.34846997|0.02528239|
|train|0|wasp|6|0.26489783|N/A|0.025033044|
|train|0|spotted toadstool|6|0.23302243|0.34779932|0.024782241|
|train|0|tree|148|0.12583475|N/A|0.025023469|
|train|0|explorer|2|0.2051028|N/A|0.025746136|
|validation|0|ground|0|N/A|N/A|0.024812693|
|validation|0|apple|6|0.19168741|0.34865176|0.025304531|
|validation|0|pear|6|0.38154968|N/A|0.025470073|
|validation|0|grapes|2|0.15973313|N/A|0.025607157|
|validation|0|leaf|6|0.23198212|N/A|0.02503063|
|validation|0|flower|6|0.19618448|N/A|0.025148346|
|validation|0|beetle|6|0.17956902|N/A|0.025148182|
|validation|0|striped mushroom|6|0.23301927|0.34779452|0.02477637|
|validation|0|spotted apple|6|0.1915674|0.34845282|0.025277897|
|validation|0|wasp|6|0.26489628|N/A|0.025029102|
|validation|0|spotted toadstool|6|0.23302527|0.34780444|0.024781044|
|validation|0|tree|148|0.12583474|N/A|0.025023228|
|validation|0|explorer|2|0.20510217|N/A|0.025745886|
|test|0|ground|0|N/A|N/A|0.024812584|
|test|0|apple|6|0.19168921|0.34865443|0.025309244|
|test|0|pear|6|0.38156153|N/A|0.025474737|
|test|0|grapes|2|0.15973673|N/A|0.025613635|
|test|0|leaf|6|0.23198511|N/A|0.025028386|
|test|0|flower|6|0.19618209|N/A|0.025148614|
|test|0|beetle|6|0.17956795|N/A|0.025142286|
|test|0|striped mushroom|6|0.23301309|0.34778571|0.024770153|
|test|0|spotted apple|6|0.19157984|0.34847835|0.025287222|
|test|0|wasp|6|0.26489694|N/A|0.025027381|
|test|0|spotted toadstool|6|0.23302973|0.3478081|0.024778664|
|test|0|tree|148|0.12583419|N/A|0.02502319|
|test|0|explorer|2|0.20510352|N/A|0.025757776|
