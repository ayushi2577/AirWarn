/* ============ setup ============ */
const TEAL="#2dd4bf", AMBER="#fbbf24", ROSE="#fb7185", VIOLET="#a78bfa", GREY="#7d8aa0";
Chart.defaults.font.family="Inter, system-ui, sans-serif";
Chart.defaults.color="#9aa7bd";
Chart.defaults.borderColor="rgba(255,255,255,.07)";
Chart.defaults.maintainAspectRatio=false;
Chart.defaults.interaction.mode="index";
Chart.defaults.interaction.intersect=false;
Chart.defaults.plugins.tooltip.backgroundColor="rgba(10,15,28,.96)";
Chart.defaults.plugins.tooltip.borderColor="rgba(255,255,255,.15)";
Chart.defaults.plugins.tooltip.borderWidth=1;
Chart.defaults.plugins.tooltip.padding=10;

Chart.register({            // vertical reference lines
  id:"vlines",
  afterDatasetsDraw(chart,args,opts){
    if(!opts||!opts.lines) return;
    const {ctx,chartArea:{top,bottom},scales:{x}}=chart;
    opts.lines.forEach(l=>{
      if(l.x==null) return;
      const px=x.getPixelForValue(l.x);
      ctx.save(); ctx.strokeStyle=l.color; ctx.lineWidth=l.width||1.5; ctx.setLineDash(l.dash||[]);
      ctx.beginPath(); ctx.moveTo(px,top); ctx.lineTo(px,bottom); ctx.stroke();
      if(l.label){ ctx.setLineDash([]); ctx.fillStyle=l.color; ctx.font="600 11px Inter, sans-serif";
        ctx.fillText(l.label,px+5,top+13); }
      ctx.restore();
    });
  }
});

const loadJSON=p=>fetch(p).then(r=>{ if(!r.ok) throw new Error(p); return r.json(); });
const fmt=(v,d=3)=>(v==null||isNaN(v))?"n/a":Number(v).toFixed(d);
const STATE={t:null,model:null,frozen:null,onChange:null};

/* ============ page polish: reveal, count-up, nav ============ */
const io=new IntersectionObserver(es=>es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add("in"); io.unobserve(e.target);} }),{threshold:.08});
document.querySelectorAll(".reveal").forEach(el=>io.observe(el));

document.querySelectorAll("[data-count]").forEach(el=>{
  const end=parseFloat(el.dataset.count), dec=+(el.dataset.dec||0), pre=el.dataset.pre||"", suf=el.dataset.suf||"";
  const t0=performance.now(), dur=1400;
  (function step(t){
    const k=Math.min(1,(t-t0)/dur), e=1-Math.pow(1-k,3);
    el.textContent=pre+(end*e).toLocaleString(undefined,{minimumFractionDigits:dec,maximumFractionDigits:dec})+suf;
    if(k<1) requestAnimationFrame(step);
  })(t0);
});

const links=[...document.querySelectorAll("nav a")];
const secs=links.map(a=>document.querySelector(a.getAttribute("href")));
addEventListener("scroll",()=>{
  const h=document.documentElement;
  document.getElementById("progress").style.width=(h.scrollTop/(h.scrollHeight-h.clientHeight)*100)+"%";
  let cur=0; secs.forEach((s,i)=>{ if(s&&s.getBoundingClientRect().top<140) cur=i; });
  links.forEach((a,i)=>a.classList.toggle("on",i===cur));
});

/* ============ 1. threshold explorer ============ */
loadJSON("data/thresholds.json").then(initExplorer).catch(()=>{
  document.getElementById("frozen").textContent="Could not load data/thresholds.json";
});

function initExplorer(d){
  const dsSel=document.getElementById("dataset"), modelSel=document.getElementById("model"),
        slider=document.getElementById("t");
  const names=Object.keys(d.validation);
  names.forEach(n=>modelSel.add(new Option(n,n)));
  if(!d.test_2019_seen) dsSel.querySelector('option[value="test_2019_seen"]').disabled=true;
  const rows=()=>d[dsSel.value][modelSel.value];
  const pers=()=>dsSel.value==="validation"?d.persistence:d.persistence_test_2019_seen;
  const nearest=(r,t)=>r.reduce((a,b)=>Math.abs(b.threshold-t)<Math.abs(a.threshold-t)?b:a);
  STATE.frozen=d.frozen_thresholds[names[0]]; STATE.model=names[0];

  const thr=new Chart(document.getElementById("chartThreshold"),{
    type:"scatter",
    data:{datasets:[
      {label:"Precision",borderColor:TEAL,backgroundColor:TEAL,showLine:true,pointRadius:0,tension:.25,data:[]},
      {label:"Recall",borderColor:AMBER,backgroundColor:AMBER,showLine:true,pointRadius:0,tension:.25,data:[]}]},
    options:{animation:false,
      plugins:{title:{display:true,text:"Precision and recall vs threshold (drag me)",color:"#c3cddd"},
               vlines:{lines:[]},legend:{labels:{boxWidth:10}}},
      scales:{x:{type:"linear",min:.05,max:.95,title:{display:true,text:"threshold"}},
              y:{min:0,max:1}}}
  });
  const pr=new Chart(document.getElementById("chartPR"),{
    type:"scatter",
    data:{datasets:[
      {label:"Model",borderColor:TEAL,backgroundColor:TEAL,showLine:true,pointRadius:0,data:[]},
      {label:"Persistence",type:"scatter",pointStyle:"star",pointRadius:11,borderColor:ROSE,backgroundColor:ROSE,data:[]},
      {label:"Your threshold",pointRadius:7,pointHoverRadius:9,borderColor:"#fff",backgroundColor:AMBER,data:[]}]},
    options:{animation:false,
      plugins:{title:{display:true,text:"Precision-recall curve",color:"#c3cddd"},legend:{labels:{boxWidth:10}},
        tooltip:{mode:"nearest",intersect:true,callbacks:{label:c=>`${c.dataset.label}: recall ${fmt(c.parsed.x)}, precision ${fmt(c.parsed.y)}`}}},
      scales:{x:{type:"linear",min:0,max:1,title:{display:true,text:"recall"}},
              y:{min:0,max:1,title:{display:true,text:"precision"}}}}
  });

  function setT(v){
    v=Math.max(.05,Math.min(.95,Math.round(v*100)/100));
    slider.value=v; draw();
  }
  const cv=document.getElementById("chartThreshold"); let down=false;
  const fromEvent=e=>setT(thr.scales.x.getValueForPixel(e.clientX-cv.getBoundingClientRect().left));
  cv.addEventListener("pointerdown",e=>{down=true;cv.setPointerCapture(e.pointerId);fromEvent(e);});
  cv.addEventListener("pointermove",e=>{if(down)fromEvent(e);});
  cv.addEventListener("pointerup",()=>down=false);

  function draw(){
    const name=modelSel.value, t=parseFloat(slider.value), rs=rows(), r=nearest(rs,t), f=d.frozen_thresholds[name];
    STATE.t=t; STATE.model=name; STATE.frozen=f;
    document.getElementById("tval").textContent=t.toFixed(2);
    ["TP","FP","FN","TN"].forEach(k=>document.getElementById(k.toLowerCase()).textContent=r[k].toLocaleString());
    document.getElementById("prec").textContent=fmt(r.precision);
    document.getElementById("rec").textContent=fmt(r.recall);
    document.getElementById("precbar").style.width=(r.precision*100)+"%";
    document.getElementById("recbar").style.width=(r.recall*100)+"%";
    document.getElementById("frozen").textContent="Frozen threshold for this model: "+f.toFixed(3)+(Math.abs(f-t)<.006?"  (you are at it)":"");
    document.getElementById("sentence").innerHTML=
      `Of <b>${(r.TP+r.FN).toLocaleString()}</b> bad-air days, the warning catches <b>${r.TP.toLocaleString()}</b> and misses <b>${r.FN.toLocaleString()}</b>. It raises <b>${r.FP.toLocaleString()}</b> false alarms.`;
    const p=pers();
    document.getElementById("pers").textContent=`Persistence (tomorrow = today), same data: precision ${fmt(p.precision)}, recall ${fmt(p.recall)}.`;

    thr.data.datasets[0].data=rs.map(x=>({x:x.threshold,y:x.precision}));
    thr.data.datasets[1].data=rs.map(x=>({x:x.threshold,y:x.recall}));
    thr.options.plugins.vlines.lines=[
      {x:f,color:AMBER,dash:[6,5],label:"frozen"},
      {x:t,color:"#ffffff",width:2}];
    thr.update("none");
    pr.data.datasets[0].label=name; pr.data.datasets[0].data=rs.map(x=>({x:x.recall,y:x.precision}));
    pr.data.datasets[1].data=[{x:p.recall,y:p.precision}];
    pr.data.datasets[2].data=[{x:r.recall,y:r.precision}];
    pr.update("none");
    if(STATE.onChange) STATE.onChange();
  }
  modelSel.onchange=draw; dsSel.onchange=draw; slider.oninput=draw;
  document.getElementById("snap").onclick=()=>setT(d.frozen_thresholds[modelSel.value]);
  slider.value=d.frozen_thresholds[names[0]]; draw();
}

/* ============ 2. final test explorer ============ */
loadJSON("data/final_results.json").then(rows=>{
  const parts=[...new Set(rows.map(r=>r.part))];
  const order=["Logistic regression","RBF SVM (Platt)","Persistence"];
  const colors=[TEAL,VIOLET,GREY];
  const tabs=document.getElementById("tabs");
  const chart=new Chart(document.getElementById("chartFinal"),{
    type:"bar",
    data:{labels:["Precision","Recall","F1","PR-AUC","Flip-day recall"],datasets:[]},
    options:{plugins:{legend:{labels:{boxWidth:10}},
        tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${fmt(c.parsed.y)}`}}},
      scales:{y:{min:0,max:1}}}
  });
  function show(p){
    [...tabs.children].forEach(b=>b.classList.toggle("on",b.dataset.p===p));
    const rs=order.map(m=>rows.find(r=>r.part===p&&r.model===m)).filter(Boolean);
    chart.data.datasets=rs.map((r,i)=>({label:r.model,backgroundColor:colors[order.indexOf(r.model)],borderRadius:6,
      data:[r.precision,r.recall,r.f1,r.pr_auc,r.change_day_recall]}));
    chart.update();
    const pos=rs[0].positives;
    let h=`<tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>PR-AUC</th><th>Flip-day recall</th><th>Precision 95% CI</th><th>Recall 95% CI</th></tr>`;
    rs.forEach(r=>{
      h+=`<tr class="${r.model==="Persistence"?"pers":""}"><td>${r.model}</td><td>${fmt(r.precision)}</td><td>${fmt(r.recall)}</td><td>${fmt(r.f1)}</td><td>${fmt(r.pr_auc)}</td><td>${fmt(r.change_day_recall)}</td><td>${r.precision_CI||"n/a"}</td><td>${r.recall_CI||"n/a"}</td></tr>`;
    });
    document.getElementById("finaltable").innerHTML=h+
      `<caption class="small" style="caption-side:bottom;text-align:left">${rs[0].rows.toLocaleString()} rows, ${pos.toLocaleString()} bad days${pos<30?' <span class="warn">low reliability: fewer than 30 bad days, no intervals</span>':""}</caption>`;
  }
  parts.forEach(p=>{
    const b=document.createElement("button"); b.dataset.p=p;
    b.textContent=p.replace("SEEN cities (headline)","seen (headline)").replace("NEW cities","new cities");
    b.onclick=()=>show(p); tabs.appendChild(b);
  });
  show(parts.find(p=>p.includes("headline"))||parts[0]);
}).catch(()=>{});

/* ============ 3. per-city ============ */
loadJSON("data/per_city.json").then(d=>{
  const dsSel=document.getElementById("citydata"), metSel=document.getElementById("citymetric");
  let cur=[];
  const chart=new Chart(document.getElementById("chartCity"),{
    type:"bar",
    data:{labels:[],datasets:[{data:[],borderRadius:5}]},
    options:{indexAxis:"y",
      plugins:{legend:{display:false},
        tooltip:{callbacks:{label:c=>`${metSel.value}: ${fmt(c.parsed.x)}`,
          afterLabel:c=>{const r=cur[c.dataIndex];return `bad days: ${r.positives}, rows: ${r.rows}`+(r.reliable?"":" (unreliable)");}}}},
      scales:{x:{min:0,max:1},y:{ticks:{autoSkip:false}}}}
  });
  function draw(){
    cur=d[dsSel.value].filter(r=>r.positives>0).sort((a,b)=>b.positives-a.positives);
    chart.data.labels=cur.map(r=>`${r.city} (${r.positives})`);
    chart.data.datasets[0].data=cur.map(r=>r[metSel.value]);
    chart.data.datasets[0].backgroundColor=cur.map(r=>r.reliable?(metSel.value==="recall"?TEAL:VIOLET):"rgba(125,138,160,.35)");
    chart.update();
  }
  dsSel.onchange=draw; metSel.onchange=draw; draw();
}).catch(()=>{});

/* ============ 4. ablation ============ */
loadJSON("data/ablation.json").then(rows=>{
  const base=rows[0].mean_pr_auc;
  const items=rows.slice(1).map(r=>({...r,delta:r.mean_pr_auc-base})).sort((a,b)=>a.delta-b.delta);
  new Chart(document.getElementById("chartAblation"),{
    type:"bar",
    data:{labels:items.map(r=>r.setting),
      datasets:[{data:items.map(r=>r.delta),borderRadius:5,
        backgroundColor:items.map(r=>r.delta<-0.005?AMBER:"rgba(125,138,160,.6)")}]},
    options:{indexAxis:"y",
      plugins:{legend:{display:false},vlines:{lines:[{x:-0.010,color:ROSE,dash:[5,5],label:"fold spread"}]},
        tooltip:{callbacks:{label:c=>`change in PR-AUC: ${c.parsed.x.toFixed(4)}`,
          afterLabel:c=>`PR-AUC ${items[c.dataIndex].mean_pr_auc.toFixed(4)} (all features ${base.toFixed(4)})\nprecision at 80% recall ${items[c.dataIndex].pooled_precision.toFixed(3)}`}}},
      scales:{x:{title:{display:true,text:"change in mean PR-AUC vs all features"}}}}
  });
}).catch(()=>{});

/* ============ 5. regularization ============ */
loadJSON("data/regularization.json").then(d=>{
  const Cs=[...new Set(d.curve.map(r=>r.C))].sort((a,b)=>a-b);
  const get=(p,c)=>(d.curve.find(r=>r.penalty===p&&r.C===c)||{});
  new Chart(document.getElementById("chartReg"),{
    type:"line",
    data:{labels:Cs.map(String),datasets:[
      {label:"L2 (Ridge)",data:Cs.map(c=>get("l2",c).mean_pr_auc),borderColor:TEAL,backgroundColor:TEAL,tension:.3,pointRadius:4},
      {label:"L1 (Lasso)",data:Cs.map(c=>get("l1",c).mean_pr_auc),borderColor:AMBER,backgroundColor:AMBER,tension:.3,pointRadius:4}]},
    options:{plugins:{legend:{labels:{boxWidth:10}},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.parsed.y.toFixed(4)}`}}},
      scales:{y:{title:{display:true,text:"mean PR-AUC (4 folds)"}},x:{title:{display:true,text:"C (larger = weaker penalty)"}}}}
  });
}).catch(()=>{});

/* ============ 6. reliability ============ */
loadJSON("data/reliability.json").then(d=>{
  const cols={"Logistic regression":TEAL,"RBF SVM (Platt)":VIOLET};
  new Chart(document.getElementById("chartRel"),{
    type:"scatter",
    data:{datasets:[
      ...Object.keys(d).map(n=>({label:n,showLine:true,borderColor:cols[n],backgroundColor:cols[n],pointRadius:4,
        data:d[n].map(p=>({x:p.pred,y:p.actual}))})),
      {label:"Perfect",showLine:true,borderColor:"rgba(255,255,255,.4)",borderDash:[5,5],pointRadius:0,data:[{x:0,y:0},{x:1,y:1}]}]},
    options:{plugins:{legend:{labels:{boxWidth:10}},tooltip:{mode:"nearest",intersect:true,
        callbacks:{label:c=>`${c.dataset.label}: predicted ${fmt(c.parsed.x,2)}, actual ${fmt(c.parsed.y,2)}`}}},
      scales:{x:{min:0,max:1,title:{display:true,text:"predicted probability"}},
              y:{min:0,max:1,title:{display:true,text:"actual share of bad days"}}}}
  });
}).catch(()=>{});

/* ============ 7. errors by month ============ */
loadJSON("data/errors_by_month.json").then(rows=>{
  const m=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  new Chart(document.getElementById("chartMonth"),{
    data:{labels:rows.map(r=>m[r.month-1]),datasets:[
      {type:"bar",label:"Bad days",data:rows.map(r=>r.bad_days),backgroundColor:"rgba(45,212,191,.35)",borderRadius:5,yAxisID:"y"},
      {type:"bar",label:"Missed",data:rows.map(r=>r.FN),backgroundColor:ROSE,borderRadius:5,yAxisID:"y"},
      {type:"bar",label:"False alarms",data:rows.map(r=>r.FP),backgroundColor:AMBER,borderRadius:5,yAxisID:"y"},
      {type:"line",label:"Recall",data:rows.map(r=>r.recall),borderColor:"#fff",backgroundColor:"#fff",tension:.3,yAxisID:"y2"}]},
    options:{plugins:{legend:{labels:{boxWidth:10}}},
      scales:{y:{title:{display:true,text:"days"}},
              y2:{position:"right",min:0,max:1,grid:{drawOnChartArea:false},title:{display:true,text:"recall"}}}}
  });
}).catch(()=>{});

/* ============ 8. city timeline ============ */
loadJSON("data/timeline.json").then(tl=>{
  const sel=document.getElementById("tlcity"), use=document.getElementById("tluse");
  const names=Object.keys(tl.cities).filter(c=>tl.cities[c].y.some(v=>v===1))
    .sort((a,b)=>tl.cities[b].y.reduce((s,v)=>s+v,0)-tl.cities[a].y.reduce((s,v)=>s+v,0));
  names.forEach(c=>sel.add(new Option(c+(tl.cities[c].seen?"":" (new city)"),c)));
  const chart=new Chart(document.getElementById("chartTimeline"),{
    type:"bar",
    data:{labels:[],datasets:[
      {type:"bar",label:"Actually bad next day",data:[],backgroundColor:"rgba(45,212,191,.16)",
        barPercentage:1,categoryPercentage:1,borderWidth:0,order:3},
      {type:"line",label:"Predicted probability",data:[],borderColor:"rgba(231,236,245,.8)",borderWidth:1.5,
        pointRadius:[],pointBackgroundColor:[],pointBorderWidth:0,tension:.15,order:1},
      {type:"line",label:"Threshold",data:[],borderColor:AMBER,borderDash:[6,5],borderWidth:1.5,pointRadius:0,order:2}]},
    options:{plugins:{legend:{labels:{boxWidth:10}},
        tooltip:{callbacks:{title:i=>"Prediction day "+i[0].label,
          label:c=>c.datasetIndex===1?`probability ${fmt(c.parsed.y,2)}`:(c.datasetIndex===0?(c.parsed.y?"actually bad next day":"actually fine next day"):`threshold ${fmt(c.parsed.y,2)}`)}}},
      scales:{y:{min:0,max:1,title:{display:true,text:"predicted probability of a bad day tomorrow"}},
              x:{ticks:{maxTicksLimit:12,callback(v){return this.getLabelForValue(v).slice(5);}}}}}
  });
  function draw(){
    const c=tl.cities[sel.value];
    const t=(use.checked&&STATE.t!=null)?STATE.t:tl.threshold;
    let tp=0,fn=0,fp=0;
    const colr=[],rad=[];
    c.p.forEach((p,i)=>{
      const pred=p>=t, y=c.y[i];
      if(y&&pred){tp++;colr.push(TEAL);rad.push(3);}
      else if(y&&!pred){fn++;colr.push(ROSE);rad.push(4);}
      else if(!y&&pred){fp++;colr.push(AMBER);rad.push(4);}
      else{colr.push("transparent");rad.push(0);}
    });
    chart.data.labels=c.dates;
    chart.data.datasets[0].data=c.y;
    chart.data.datasets[1].data=c.p;
    chart.data.datasets[1].pointBackgroundColor=colr;
    chart.data.datasets[1].pointRadius=rad;
    chart.data.datasets[2].data=c.dates.map(()=>t);
    chart.update("none");
    const bad=tp+fn;
    document.getElementById("tlsummary").innerHTML=
      `<b>${sel.value}</b>${c.seen?"":" (a city never seen in training)"}: ${bad} bad days in 2019; caught <b>${tp}</b>, missed <b>${fn}</b>, false alarms <b>${fp}</b> at threshold ${t.toFixed(2)}.`;
  }
  sel.onchange=draw; use.onchange=draw;
  const prev=STATE.onChange; STATE.onChange=()=>{ if(prev) prev(); if(use.checked) draw(); };
  draw();
}).catch(()=>{ document.getElementById("tlsummary").textContent="Run src/export_site_data.py to create data/timeline.json"; });

/* expandable rows: click anywhere on the row header */
document.addEventListener("click",e=>{
  const head=e.target.closest(".acc .head"); if(!head) return;
  const row=head.parentElement, open=row.classList.toggle("open");
  const b=head.querySelector(".chev"); if(b) b.setAttribute("aria-expanded",open);
});

function initExplorer(d){
  const dsSel=document.getElementById("dataset"), modelSel=document.getElementById("model"),
        slider=document.getElementById("t");
  const names=Object.keys(d.validation);
  names.forEach(n=>modelSel.add(new Option(n,n)));
  if(!d.test_2019_seen) dsSel.querySelector('option[value="test_2019_seen"]').disabled=true;
  const rows=()=>d[dsSel.value][modelSel.value];
  const pers=()=>dsSel.value==="validation"?d.persistence:d.persistence_test_2019_seen;
  const nearest=(r,t)=>r.reduce((a,b)=>Math.abs(b.threshold-t)<Math.abs(a.threshold-t)?b:a);
  STATE.frozen=d.frozen_thresholds[names[0]]; STATE.model=names[0];

  /* 2. precision and recall vs threshold, with live dots on both curves */
  const thr=new Chart(document.getElementById("chartThreshold"),{
    type:"scatter",
    data:{datasets:[
      {label:"Precision",borderColor:TEAL,backgroundColor:TEAL,showLine:true,pointRadius:0,tension:.25,data:[]},
      {label:"Recall",borderColor:AMBER,backgroundColor:AMBER,showLine:true,pointRadius:0,tension:.25,data:[]},
      {label:"Precision here",borderColor:"#fff",backgroundColor:TEAL,pointRadius:8,data:[]},
      {label:"Recall here",borderColor:"#fff",backgroundColor:AMBER,pointRadius:8,data:[]}]},
    options:{animation:false,
      plugins:{vlines:{lines:[]},legend:{labels:{boxWidth:10,filter:i=>i.datasetIndex<2}},
        tooltip:{filter:i=>i.datasetIndex<2}},
      scales:{x:{type:"linear",min:.05,max:.95,title:{display:true,text:"threshold"}},
              y:{min:0,max:1}}}
  });

  /* 3. precision-recall curve, with guide lines from the dot to both axes */
  const pr=new Chart(document.getElementById("chartPR"),{
    type:"scatter",
    data:{datasets:[
      {label:"Model",borderColor:TEAL,backgroundColor:TEAL,showLine:true,pointRadius:0,data:[]},
      {label:"Persistence",pointStyle:"star",pointRadius:11,borderColor:ROSE,backgroundColor:ROSE,data:[]},
      {label:"Your threshold",pointRadius:9,pointHoverRadius:11,borderColor:"#fff",backgroundColor:AMBER,data:[]},
      {label:"g1",showLine:true,borderColor:"rgba(251,191,36,.7)",borderDash:[5,5],borderWidth:1.5,pointRadius:0,data:[]},
      {label:"g2",showLine:true,borderColor:"rgba(251,191,36,.7)",borderDash:[5,5],borderWidth:1.5,pointRadius:0,data:[]}]},
    options:{animation:false,
      plugins:{legend:{labels:{boxWidth:10,filter:i=>i.datasetIndex<3}},
        tooltip:{mode:"nearest",intersect:true,filter:i=>i.datasetIndex<3,
          callbacks:{label:c=>`${c.dataset.label}: recall ${fmt(c.parsed.x)}, precision ${fmt(c.parsed.y)}`}}},
      scales:{x:{type:"linear",min:0,max:1,title:{display:true,text:"recall"}},
              y:{min:0,max:1,title:{display:true,text:"precision"}}}}
  });

  /* 4. outcome bars: caught vs missed, right warnings vs false alarms */
  const out=new Chart(document.getElementById("chartOutcome"),{
    type:"bar",
    data:{labels:["Real bad days","Warnings issued"],datasets:[
      {label:"Caught",backgroundColor:TEAL,borderRadius:4,data:[0,0]},
      {label:"Missed",backgroundColor:ROSE,borderRadius:4,data:[0,0]},
      {label:"False alarms",backgroundColor:AMBER,borderRadius:4,data:[0,0]}]},
    options:{indexAxis:"y",animation:{duration:150},
      plugins:{legend:{labels:{boxWidth:10}}},
      scales:{x:{stacked:true,title:{display:true,text:"number of days"}},
              y:{stacked:true}}}
  });

  function setT(v){
    v=Math.max(.05,Math.min(.95,Math.round(v*100)/100));
    slider.value=v; draw();
  }
  const cv=document.getElementById("chartThreshold"); let down=false;
  const fromEvent=e=>setT(thr.scales.x.getValueForPixel(e.clientX-cv.getBoundingClientRect().left));
  cv.addEventListener("pointerdown",e=>{down=true;cv.setPointerCapture(e.pointerId);fromEvent(e);});
  cv.addEventListener("pointermove",e=>{if(down)fromEvent(e);});
  cv.addEventListener("pointerup",()=>down=false);

  function draw(){
    const name=modelSel.value, t=parseFloat(slider.value), rs=rows(), r=nearest(rs,t), f=d.frozen_thresholds[name];
    STATE.t=t; STATE.model=name; STATE.frozen=f;
    document.getElementById("tval").textContent=t.toFixed(2);
    ["TP","FP","FN","TN"].forEach(k=>document.getElementById(k.toLowerCase()).textContent=r[k].toLocaleString());
    document.getElementById("prec").textContent=fmt(r.precision);
    document.getElementById("rec").textContent=fmt(r.recall);
    document.getElementById("precbar").style.width=(r.precision*100)+"%";
    document.getElementById("recbar").style.width=(r.recall*100)+"%";
    document.getElementById("frozen").textContent="Frozen threshold for this model: "+f.toFixed(3)+(Math.abs(f-t)<.006?"  (you are at it)":"");
    document.getElementById("sentence").innerHTML=
      `Of <b>${(r.TP+r.FN).toLocaleString()}</b> bad-air days, the warning catches <b>${r.TP.toLocaleString()}</b> and misses <b>${r.FN.toLocaleString()}</b>. It raises <b>${r.FP.toLocaleString()}</b> false alarms.`;
    const p=pers();
    document.getElementById("pers").textContent=`Persistence (tomorrow = today), same data: precision ${fmt(p.precision)}, recall ${fmt(p.recall)}.`;

    thr.data.datasets[0].data=rs.map(x=>({x:x.threshold,y:x.precision}));
    thr.data.datasets[1].data=rs.map(x=>({x:x.threshold,y:x.recall}));
    thr.data.datasets[2].data=[{x:r.threshold,y:r.precision}];
    thr.data.datasets[3].data=[{x:r.threshold,y:r.recall}];
    thr.options.plugins.vlines.lines=[
      {x:f,color:AMBER,dash:[6,5],label:"frozen"},
      {x:r.threshold,color:"#ffffff",width:2}];
    thr.update("none");

    pr.data.datasets[0].label=name;
    pr.data.datasets[0].data=rs.map(x=>({x:x.recall,y:x.precision}));
    pr.data.datasets[1].data=[{x:p.recall,y:p.precision}];
    pr.data.datasets[2].data=[{x:r.recall,y:r.precision}];
    pr.data.datasets[3].data=[{x:r.recall,y:0},{x:r.recall,y:r.precision}];
    pr.data.datasets[4].data=[{x:0,y:r.precision},{x:r.recall,y:r.precision}];
    pr.update("none");

    out.data.datasets[0].data=[r.TP,r.TP];
    out.data.datasets[1].data=[r.FN,0];
    out.data.datasets[2].data=[0,r.FP];
    out.update();

    if(STATE.onChange) STATE.onChange();
  }
  modelSel.onchange=draw; dsSel.onchange=draw; slider.oninput=draw;
  document.getElementById("snap").onclick=()=>setT(d.frozen_thresholds[modelSel.value]);
  slider.value=d.frozen_thresholds[names[0]]; draw();
}