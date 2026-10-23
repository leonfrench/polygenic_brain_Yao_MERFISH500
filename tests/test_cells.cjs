// Run with: node tests/test_cells.cjs
const assert=require('node:assert/strict');
const R=require('../spatial_app/cell_component/cells.js');
// Packed colours are little-endian RGBA for a Uint32Array view of ImageData.
const packed=new Uint8Array(new Uint32Array([R.pack([1,2,3])]).buffer);assert.deepEqual([...packed],[1,2,3,255]);
assert.deepEqual(R.gradient(0),R.rgb(R.STOPS[0]));assert.deepEqual(R.gradient(1),R.rgb(R.STOPS[4]));
assert.deepEqual(R.limits([NaN,.3,.7]),[.3,.7]);assert.equal(R.limits([NaN]),null);const flat=R.limits([.5,.5]);assert(Math.abs(flat[0]-.495)<1e-12&&Math.abs(flat[1]-.505)<1e-12);
// Percentile clipping: 5th/95th of 0..100 are 5 and 95; outliers and NaN do not move the limits.
assert.deepEqual(R.clippedLimits(Array.from({length:101},(_,i)=>i)),[5,95]);
const skewed=new Float64Array(101).map((_,i)=>i);skewed[100]=1e6;skewed[0]=NaN;
const [lo,hi]=R.clippedLimits(skewed);assert(Math.abs(lo-5.95)<1e-9&&Math.abs(hi-95.05)<1e-9);// numpy.quantile gives the same
assert.equal(R.clippedLimits([NaN]),null);assert.deepEqual(R.clippedLimits([.4,.4,.4]).map(v=>+v.toFixed(3)),[.395,.405]);
// Per-slice limits: each group gets its own percentiles; an unscored group gets null.
const values=new Float64Array(202),groups=new Uint8Array(202);for(let i=0;i<101;i++){values[i]=i;values[101+i]=1000+10*i;groups[101+i]=1;}
values[0]=NaN;const per=R.groupLimits(values,groups,3);
assert.deepEqual(per[1],[1050,1950]);assert(Math.abs(per[0][0]-5.95)<1e-9);assert.equal(per[2],null);
// Discs are symmetric, never wider than their diameter, and small sizes are solid squares.
assert.deepEqual(R.disc(2),[[0,0,2],[1,0,2]]);
for(const size of [4,7,12]){const spans=R.disc(size);for(const [y,a,b] of spans){assert(a>=0&&b<=size);const mirror=spans.find(s=>s[0]===size-1-y);assert.deepEqual([mirror[1],mirror[2]],[a,b]);}}
const buf=new Uint32Array(25);R.stamp(buf,5,5,-1,3,R.disc(3),9);assert.equal(buf.filter(v=>v===9).length,4);// clipped at the edges
// Nearest-cell lookup across bucket boundaries, with quantization step and radius.
const px=new Uint16Array([10,50,51,400]),py=new Uint16Array([10,50,49,400]);
const index=R.buildIndex(px,py,2,1000,1000,64);
assert.equal(R.nearest(index,px,py,99,99,30),1);assert.equal(R.nearest(index,px,py,103,97,30),2);
assert.equal(R.nearest(index,px,py,500,500,30),-1);assert.equal(R.nearest(index,px,py,790,810,30),3);
console.log('Cell viewer colour, disc, clipping and nearest-cell tests passed');
