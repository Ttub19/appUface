import test from "node:test";
import assert from "node:assert/strict";
import { GoldenRatioCalc, GoldenRatioStabilizer } from "../js/goldenRatioCalc.js";
import { ArRenderer, ArTransformSmoother, GLASSES_FACE_FIT_RATIO } from "../js/arRenderer.js";
import { LM } from "../js/landmarks.js";
const PHI=(1+Math.sqrt(5))/2;
function face(){const lm=Array.from({length:478},()=>({x:.5,y:.5,z:0}));const p=(i,x,y)=>lm[i]={x,y,z:0};const hw=(.6/PHI)/2;p(LM.FOREHEAD,.5,.1);p(LM.GLABELLA,.5,.3);p(LM.NOSE_BRIDGE,.5,.36);p(LM.NOSE_TIP,.5,.44);p(LM.NOSE_BASE,.5,.5);p(LM.CHIN,.5,.7);p(LM.CHEEK_L,.5-hw,.43);p(LM.CHEEK_R,.5+hw,.43);p(LM.EYE_L_OUTER,.35,.35);p(LM.EYE_L_INNER,.43,.35);p(LM.EYE_R_INNER,.57,.35);p(LM.EYE_R_OUTER,.65,.35);p(LM.BROW_L_OUTER,.35,.28);p(LM.BROW_L_INNER,.44,.28);p(LM.BROW_R_INNER,.56,.28);p(LM.BROW_R_OUTER,.65,.28);p(LM.MOUTH_L,.43,.56);p(LM.MOUTH_R,.57,.56);p(LM.JAW_L,.35,.63);p(LM.JAW_R,.65,.63);p(LM.IRIS_A,.39,.35);p(LM.IRIS_B,.61,.35);return lm;}
function rotate(lm,deg){const a=deg*Math.PI/180,c=Math.cos(a),s=Math.sin(a);return lm.map(p=>{const x=p.x-.5,y=p.y-.4;return {...p,x:.5+x*c-y*s,y:.4+x*s+y*c};});}
test("ideal geometry",()=>{const r=GoldenRatioCalc.compute(face(),1000,1000);assert.equal(r.quality.isUsable,true);assert.ok(Math.abs(r.ratio-PHI)<1e-9);assert.ok(r.overall>99.9);});
test("roll alignment",()=>{const r=GoldenRatioCalc.compute(rotate(face(),15),1000,1000);assert.equal(r.quality.isUsable,true);assert.ok(Math.abs(r.ratio-PHI)<1e-8);});
test("yaw gate",()=>{const f=face();f[LM.NOSE_TIP].x=.62;const r=GoldenRatioCalc.compute(f,1000,1000);assert.ok(r.quality.reasons.includes("yaw"));});
test("invalid input",()=>{const f=face();f[LM.CHIN].x=NaN;assert.throws(()=>GoldenRatioCalc.compute(f,1000,1000));});
test("stable window",()=>{const s=new GoldenRatioStabilizer({minSamples:3});const m=GoldenRatioCalc.compute(face(),1000,1000);s.push(m);s.push(m);assert.equal(s.push(m).stable,true);});
test("eye center transform",()=>{const r=ArRenderer.getEyeTransform(face().slice(0,468),1000,1000);assert.ok(Math.abs(r.cx-500)<1e-9);assert.ok(Math.abs(r.span-220)<1e-9);});
test("face width from temples",()=>{const f=face();f[LM.TEMPLE_L]={x:.25,y:.35,z:0};f[LM.TEMPLE_R]={x:.75,y:.35,z:0};const r=ArRenderer.getEyeTransform(f,1000,1000);assert.ok(Math.abs(r.faceWidth-500)<1e-9);});
test("face width falls back when temples implausible",()=>{const r=ArRenderer.getEyeTransform(face(),1000,1000);assert.ok(Math.abs(r.faceWidth-220*1.85)<1e-9);});
test("glasses fit spans face width",()=>{const t={span:220,faceWidth:500};const k=GLASSES_FACE_FIT_RATIO;assert.ok(Math.abs(ArRenderer.fitGlassesScale(t,{scale:1,frameWidth:250})*250-500*k)<1e-9);assert.ok(Math.abs(ArRenderer.fitGlassesScale(t,{scale:1,frameWidth:250},1.1)*250-550*k)<1e-9);assert.ok(Math.abs(ArRenderer.fitGlassesScale(t,{scale:1})-220/180)<1e-9);});
test("smoother tracks face width",()=>{const s=new ArTransformSmoother({timeConstantMs:50});s.update({cx:0,cy:0,span:1,angle:0,faceWidth:100},0);const r=s.update({cx:0,cy:0,span:1,angle:0,faceWidth:200},50);assert.ok(r.faceWidth>100&&r.faceWidth<200);});
test("angle smoothing wraps",()=>{const s=new ArTransformSmoother({timeConstantMs:50});s.update({cx:0,cy:0,span:1,angle:Math.PI-.02},0);assert.ok(Math.abs(s.update({cx:1,cy:1,span:2,angle:-Math.PI+.02},50).angle)>3);});
