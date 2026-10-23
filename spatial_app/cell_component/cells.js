/* Pure numeric helpers for the cell viewer, also exercised by the Node tests. */
(function(root) {
const STOPS=['#32265C','#315B91','#23958B','#7ECF8D','#F3DD65'];
function rgb(hex){return [1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));}
// Little-endian RGBA packed for a Uint32Array view of ImageData.
function pack(c,alpha=255){return ((alpha<<24)|(c[2]<<16)|(c[1]<<8)|c[0])>>>0;}
const stops=STOPS.map(rgb);
function gradient(t){t=Math.max(0,Math.min(1,t))*4;const i=Math.min(3,Math.floor(t)),f=t-i;return stops[i].map((x,j)=>Math.round(x+(stops[i+1][j]-x)*f));}
function limits(values){let low=Infinity,high=-Infinity;for(const v of values)if(Number.isFinite(v)){if(v<low)low=v;if(v>high)high=v;}
 if(!Number.isFinite(low))return null;if(low===high){low=Math.max(0,low-.005);high=Math.min(1,high+.005);}return [low,high];}
// Colour limits at the q and 1-q quantiles of the finite values (linear interpolation, as numpy).
// Values beyond them take the end colours.
function clippedLimits(values,q=.05){let n=0;const finite=new Float64Array(values.length);for(const v of values)if(Number.isFinite(v))finite[n++]=v;
 if(!n)return null;const sorted=finite.subarray(0,n).sort();
 const at=p=>{const pos=(n-1)*p,i=Math.floor(pos),f=pos-i;return i+1<n?sorted[i]+(sorted[i+1]-sorted[i])*f:sorted[i];};
 return limits([at(q),at(1-q)]);}
// clippedLimits within each group (slice): [low, high] per group, or null when a group has no finite value.
function groupLimits(values,groups,n,q=.05){const count=new Int32Array(n),fill=new Int32Array(n);
 for(let i=0;i<values.length;i++)if(Number.isFinite(values[i]))count[groups[i]]++;
 const buckets=Array.from(count,c=>new Float64Array(c));
 for(let i=0;i<values.length;i++){const v=values[i];if(Number.isFinite(v)){const k=groups[i];buckets[k][fill[k]++]=v;}}
 return buckets.map(b=>clippedLimits(b,q));}
// Rows of a filled disc of the given pixel diameter: [dy, x0, x1) offsets from the top-left corner.
function disc(size){const spans=[];if(size<=3){for(let y=0;y<size;y++)spans.push([y,0,size]);return spans;}
 const r=size/2;for(let y=0;y<size;y++){const dy=y+.5-r,half=Math.sqrt(Math.max(0,r*r-dy*dy));const x0=Math.round(r-half),x1=Math.round(r+half);if(x1>x0)spans.push([y,x0,x1]);}return spans;}
function stamp(buf,W,H,x,y,spans,color){for(const [dy,a,b] of spans){const yy=y+dy;if(yy<0||yy>=H)continue;const lo=Math.max(0,x+a),hi=Math.min(W,x+b);if(hi>lo)buf.fill(color,yy*W+lo,yy*W+hi);}}
// Uniform bucket index over display coordinates for nearest-cell hover lookup.
function buildIndex(px,py,step,width,height,bucket){const bw=Math.max(1,Math.ceil(width/bucket)),bh=Math.max(1,Math.ceil(height/bucket)),n=px.length;
 const key=new Int32Array(n),start=new Int32Array(bw*bh+1);
 for(let i=0;i<n;i++){const k=Math.min(bh-1,Math.floor(py[i]*step/bucket))*bw+Math.min(bw-1,Math.floor(px[i]*step/bucket));key[i]=k;start[k+1]++;}
 for(let k=0;k<bw*bh;k++)start[k+1]+=start[k];
 const fill=start.slice(0,bw*bh),items=new Int32Array(n);for(let i=0;i<n;i++)items[fill[key[i]]++]=i;
 return {bw,bh,bucket,start,items,step};}
function nearest(index,px,py,x,y,radius){const {bw,bh,bucket,start,items,step}=index;let best=-1,bestD=radius*radius;
 const reach=Math.ceil(radius/bucket),cx=Math.floor(x/bucket),cy=Math.floor(y/bucket);
 for(let by=Math.max(0,cy-reach);by<=Math.min(bh-1,cy+reach);by++)for(let bx=Math.max(0,cx-reach);bx<=Math.min(bw-1,cx+reach);bx++){const k=by*bw+bx;
  for(let j=start[k];j<start[k+1];j++){const i=items[j],dx=px[i]*step-x,dy=py[i]*step-y,d=dx*dx+dy*dy;if(d<bestD||(d===bestD&&i<best)){bestD=d;best=i;}}}
 return best;}
function precision(low,high){return Math.min(8,Math.max(3,Math.ceil(-Math.log10((high-low)/4||1e-3))+1));}
// Hover levels: the configured keys that exist, plus each extra key (the sidebar's levels) not already
// listed, placed before the first listed level of the same kind, so a coarser class or division comes first.
function hoverKeys(levels,keys,extra){let out=keys.filter(k=>levels[k]);
 for(const k of extra){if(!levels[k]||out.includes(k))continue;const at=out.findIndex(x=>levels[x].kind===levels[k].kind);
  out=at<0?[...out,k]:[...out.slice(0,at),k,...out.slice(at)];}
 return out;}
// One line per level for cell i: "Title: name (label)" when the label has a description (region acronyms),
// otherwise "Title: label"; the sliceWith level's line ends with the cell's slice.
function hoverLines(levels,keys,i,slice,sliceWith){return keys.map(k=>{const lvl=levels[k],c=lvl.codes[i],label=lvl.labels[c],d=lvl.descriptions[c];
 const text=lvl.title+': '+(d&&d!==label?d+' ('+label+')':label);return k===sliceWith?text+' · slice: '+slice:text;});}
root.CellRender={rgb,pack,gradient,limits,clippedLimits,groupLimits,disc,stamp,buildIndex,nearest,precision,hoverKeys,hoverLines,STOPS};
if(typeof module!=='undefined')module.exports=root.CellRender;
})(globalThis);
