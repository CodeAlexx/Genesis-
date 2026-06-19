// fpx_gpu.c — OpenCL compute shim for the MojoMedia editor's GPU pipeline.
//
// Replaces the Mojo/MAX (LayoutTensor + DeviceContext + enqueue_function) kernels so
// the editor compiles host-only (no MAX device-codegen → ~5min/60GB build collapses).
// The kernels are RUNTIME-compiled by the OpenCL driver (clBuildProgram at init), so
// there is ZERO GPU codegen in the Mojo/C build AND it runs on any OpenCL device
// (NVIDIA/AMD/Intel) — keeping the portability MAX gave us.
//
// Math is ported 1:1 from editor/mojo_max_reference/{main_editor_max,transitions_max}.mojo.
// Flat index matches Mojo's Layout.row_major(VH,VW,4):  i = (y*VW + x)*4 + c.
//
// Pipeline (one upload, chain on-device, one download — mirrors the Mojo flow):
//   upload base/over/trans (u8 -> f32 on GPU)
//   track1 = transition(base,trans,t,param)  OR  base           (fpx_gpu_track1)
//   in     = composite_pip(track1, over, op, px,py,pw,ph)        (fpx_gpu_pip)
//   mid    = brightness(in);  out = contrast(mid)                (fpx_gpu_grade)
//   look   = lut3d(out)|vhs(out)  (optional)                     (fpx_gpu_look)
//   download final (out|look) -> host f32 (encode) or u8 (draw)
//
// Build:  cc -O2 -fPIC -shared fpx_gpu.c -o libfpxgpu.so -I/usr/local/cuda/include -lOpenCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// fixed working resolution (matches the editor's VW/VH/vlayout)
#define GVW 1280
#define GVH 856
#define GPIX (GVW * GVH)
#define GN (GPIX * 4)
#define MAXLUTF (33 * 33 * 33 * 3)

enum { BASE = 0, OVER, TRANS, TRACK1, INB, MID, OUTB, LOOKB, NBUF };

static cl_context     g_ctx = NULL;
static cl_command_queue g_q = NULL;
static cl_program     g_prog = NULL;
static cl_device_id   g_dev = NULL;
static cl_mem g_buf[NBUF];        // VW*VH*4 floats each
static cl_mem g_lut = NULL;       // MAXLUTF floats
static cl_mem g_stage = NULL;     // VW*VH*4 bytes (u8 upload staging)
static cl_mem g_hist = NULL;      // 3*256 int histogram bins
static cl_mem g_grid = NULL;      // 256*256 int scope accumulator (waveform / vectorscope)
static cl_mem g_parade = NULL;    // 3*256*256 int scope accumulator (RGB parade: one panel per channel)
static cl_mem g_scope = NULL;     // 256*256*4 byte rendered scope image
static cl_mem g_tmp = NULL;       // VW*VH*4 float scratch (P2: transform source copy / blur ping-pong)
static cl_mem g_tmp2 = NULL;      // VW*VH*4 float scratch #2 (P9: glow bright-pass + blur partner)
static int g_ready = 0;

// every per-pixel kernel; VW/VH injected as -D build options.
static const char* KSRC =
"#define IDX(x,y) (((y)*VW+(x))*4)\n"
"float clamp01(float v){ return v<0.0f?0.0f:(v>1.0f?1.0f:v); }\n"
// u8 -> f32 [0,1]
"__kernel void k_unpack(__global const uchar* s,__global float* d){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=(float)s[i+0]/255.0f; d[i+1]=(float)s[i+1]/255.0f; d[i+2]=(float)s[i+2]/255.0f; d[i+3]=(float)s[i+3]/255.0f;\n"
"}\n"
// f32 -> u8 (round)
"__kernel void k_pack(__global const float* s,__global uchar* d){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=(uchar)(clamp01(s[i+0])*255.0f+0.5f); d[i+1]=(uchar)(clamp01(s[i+1])*255.0f+0.5f);\n"
"  d[i+2]=(uchar)(clamp01(s[i+2])*255.0f+0.5f); d[i+3]=(uchar)(clamp01(s[i+3])*255.0f+0.5f);\n"
"}\n"
"__kernel void k_copy(__global const float* s,__global float* d){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=s[i+0]; d[i+1]=s[i+1]; d[i+2]=s[i+2]; d[i+3]=s[i+3];\n"
"}\n"
// scope: RGB histogram — 3x256 int bins (R:0..255, G:256..511, B:512..767)
"__kernel void k_hist_clear(__global int* h){ int i=get_global_id(0); if(i<768) h[i]=0; }\n"
"__kernel void k_hist(__global const float* s,__global volatile int* h){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int r=(int)(clamp01(s[i+0])*255.0f+0.5f), g=(int)(clamp01(s[i+1])*255.0f+0.5f), b=(int)(clamp01(s[i+2])*255.0f+0.5f);\n"
"  atomic_inc(&h[r]); atomic_inc(&h[256+g]); atomic_inc(&h[512+b]);\n"
"}\n"
// scope: 256x256 int grid (waveform: col x value ; vectorscope: U x V) + RGBA8 renderers
"__kernel void k_grid_clear(__global int* g){ int i=get_global_id(0); if(i<65536) g[i]=0; }\n"
"__kernel void k_wave_acc(__global const float* s,__global volatile int* w){\n"   // w[col*256+val], luma waveform
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int col=x*256/VW; float luma=clamp01(s[i+0])*0.299f+clamp01(s[i+1])*0.587f+clamp01(s[i+2])*0.114f;\n"
"  int val=(int)(luma*255.0f+0.5f); if(val>255)val=255; atomic_inc(&w[col*256+val]);\n"
"}\n"
"__kernel void k_wave_img(__global const int* w,__global uchar* img,float gain){\n" // 256x256 greenish, y=value (top=bright)
"  int ix=get_global_id(0),iy=get_global_id(1); if(ix>=256||iy>=256) return; int o=(iy*256+ix)*4;\n"
"  int c=w[ix*256+(255-iy)]; float v=(float)c*gain; if(v>1.0f)v=1.0f; uchar g=(uchar)(v*255.0f);\n"
"  img[o+0]=(uchar)((float)g*0.55f); img[o+1]=g; img[o+2]=(uchar)((float)g*0.7f); img[o+3]=255;\n"
"}\n"
"__kernel void k_vec_acc(__global const float* s,__global volatile int* w){\n"  // U/V scatter (BT.601, 0..1 centred at .5)
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(s[i+0]),g=clamp01(s[i+1]),b=clamp01(s[i+2]);\n"
"  float u=-0.169f*r-0.331f*g+0.5f*b+0.5f; float v=0.5f*r-0.419f*g-0.081f*b+0.5f;\n"
"  int ux=(int)(u*255.0f+0.5f), vy=(int)(v*255.0f+0.5f); if(ux<0)ux=0; if(ux>255)ux=255; if(vy<0)vy=0; if(vy>255)vy=255;\n"
"  atomic_inc(&w[vy*256+ux]);\n"
"}\n"
"__kernel void k_vec_img(__global const int* w,__global uchar* img,float gain){\n"
"  int ix=get_global_id(0),iy=get_global_id(1); if(ix>=256||iy>=256) return; int o=(iy*256+ix)*4;\n"
"  int c=w[(255-iy)*256+ix]; float v=(float)c*gain; if(v>1.0f)v=1.0f; uchar g=(uchar)(v*255.0f);\n"
"  img[o+0]=(uchar)((float)g*0.5f); img[o+1]=g; img[o+2]=(uchar)((float)g*0.55f); img[o+3]=255;\n"
"}\n"
// RGB PARADE (Triad-B P1): per-channel column waveform, 3 side-by-side panels (R|G|B). The grid is
// channel-major p[c*65536 + col*256 + val] — for each source pixel, accumulate (column, value) per
// channel (col = source x compressed to 0..255). The image renders R into x[0,85), G into [85,170),
// B into [170,256), with value on the y-axis (top = brightest, like Shotcut's 255-value layout).
"__kernel void k_parade_acc(__global const float* s,__global volatile int* p){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int col=x*256/VW;\n"
"  int rv=(int)(clamp01(s[i+0])*255.0f+0.5f); if(rv>255)rv=255;\n"
"  int gv=(int)(clamp01(s[i+1])*255.0f+0.5f); if(gv>255)gv=255;\n"
"  int bv=(int)(clamp01(s[i+2])*255.0f+0.5f); if(bv>255)bv=255;\n"
"  atomic_inc(&p[0*65536+col*256+rv]); atomic_inc(&p[1*65536+col*256+gv]); atomic_inc(&p[2*65536+col*256+bv]);\n"
"}\n"
// 256x256 image: three 85/85/86-wide panels. For output pixel (ix,iy): pick the panel/channel by ix,
// map ix within the panel back to a source column (0..255), read the grid at value (255-iy), scale by
// gain, and paint in that channel's color. A 1px gap between panels (dark) separates the channels.
"__kernel void k_parade_img(__global const int* p,__global uchar* img,float gain){\n"
"  int ix=get_global_id(0),iy=get_global_id(1); if(ix>=256||iy>=256) return; int o=(iy*256+ix)*4;\n"
"  int ch=-1; int loc=0; int pw=85;\n"
"  if(ix<85){ ch=0; loc=ix; pw=85; }\n"
"  else if(ix<170){ ch=1; loc=ix-85; pw=85; }\n"
"  else { ch=2; loc=ix-170; pw=86; }\n"
"  // 1px dark separator at the left edge of the G and B panels.\n"
"  if((ix==85)||(ix==170)){ img[o+0]=8; img[o+1]=10; img[o+2]=14; img[o+3]=255; return; }\n"
"  int col=loc*256/pw; if(col>255)col=255;\n"
"  int val=255-iy;\n"
"  int c=p[ch*65536+col*256+val]; float v=(float)c*gain; if(v>1.0f)v=1.0f; uchar g=(uchar)(v*255.0f);\n"
"  uchar r8=0,g8=0,b8=0;\n"
"  if(ch==0){ r8=g; g8=(uchar)((float)g*0.18f); b8=(uchar)((float)g*0.18f); }\n"
"  else if(ch==1){ r8=(uchar)((float)g*0.18f); g8=g; b8=(uchar)((float)g*0.18f); }\n"
"  else { r8=(uchar)((float)g*0.3f); g8=(uchar)((float)g*0.45f); b8=g; }\n"
"  img[o+0]=r8; img[o+1]=g8; img[o+2]=b8; img[o+3]=255;\n"
"}\n"
"__kernel void k_parade_clear(__global int* p){ int i=get_global_id(0); if(i<3*65536) p[i]=0; }\n"
// composite alpha-over: dst = over*aeff + base*(1-aeff), aeff=over.a*op ; alpha from base
"__kernel void k_composite(__global const float* base,__global const float* over,__global float* dst,float op){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float a=over[i+3]*op, inv=1.0f-a;\n"
"  dst[i+0]=over[i+0]*a+base[i+0]*inv; dst[i+1]=over[i+1]*a+base[i+1]*inv; dst[i+2]=over[i+2]*a+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
// picture-in-picture: shrink whole over into normalized rect [px,py,pw,ph]. The composite weight is
// over.a * op, so a pixel whose OVER ALPHA was zeroed (e.g. by k_chroma keying out the green screen)
// contributes nothing and the base (track1) shows through — this is what makes the chroma key visible
// after compositing. (Identical to the pre-P4 behaviour when over.a==1 everywhere.)
"__kernel void k_pip(__global const float* base,__global const float* over,__global float* dst,float op,float px,float py,float pw,float ph){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float fx=(float)x/(float)VW, fy=(float)y/(float)VH;\n"
"  if(pw>0.0f && ph>0.0f && fx>=px && fx<px+pw && fy>=py && fy<py+ph){\n"
"    int sx=(int)((fx-px)/pw*(float)VW), sy=(int)((fy-py)/ph*(float)VH);\n"
"    if(sx<0)sx=0; if(sx>VW-1)sx=VW-1; if(sy<0)sy=0; if(sy>VH-1)sy=VH-1; int si=IDX(sx,sy);\n"
"    float a=over[si+3]*op, inv=1.0f-a;\n"
"    dst[i+0]=over[si+0]*a+base[i+0]*inv; dst[i+1]=over[si+1]*a+base[i+1]*inv; dst[i+2]=over[si+2]*a+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"  } else { dst[i+0]=base[i+0]; dst[i+1]=base[i+1]; dst[i+2]=base[i+2]; dst[i+3]=base[i+3]; }\n"
"}\n"
// P4 CHROMA KEY (green-screen). Runs IN PLACE on the OVER buffer BEFORE k_pip, when enabled. For each
// OVER pixel we measure a CHROMA-VECTOR distance to the key colour (kr,kg,kb): subtract each colour's
// luma (BT.601) to get its chroma vector (the colour minus its grey component), then take the
// euclidean distance between the pixel's chroma vector and the key's chroma vector. This removes the
// COMMON luma term so two colours that differ only by a flat brightness offset (same RGB ratios, e.g.
// a uniformly darkened patch) compare equal — but it is CHROMA-VECTOR PROXIMITY, not true hue: a
// DESATURATED or much darker shade of the key colour has a shorter chroma vector and so a larger
// distance, and is NOT reliably keyed (a follow-up pass could normalize the chroma vectors / compare
// their direction for hue-true keying on shaded real footage). It is gate-correct for the flat
// full-bright key-over-flat-base case and matches Shotcut's bluescreen0r frei0r 'Distance' threshold
// closely enough for that. dist is the euclidean length of the luma-removed difference, then we map it
// through ck_smoothstep(sim, sim+smth, dist): close to the key (dist<=sim) → 0 (fully keyed/
// transparent), far from the key (dist>=sim+smth) → 1 (opaque), with a soft edge band of width 'smth'.
// We ONLY scale over.a (rgb untouched), so k_pip's `over.a*op` weight then shows the base through the
// keyed pixels. NB var names avoid reserved OpenCL words (no local/global/half/etc).
"float ck_smoothstep(float e0,float e1,float v){\n"
"  float d=e1-e0; if(d<=0.0f){ return v<e0?0.0f:1.0f; }\n"
"  float tnorm=(v-e0)/d; if(tnorm<0.0f)tnorm=0.0f; if(tnorm>1.0f)tnorm=1.0f;\n"
"  return tnorm*tnorm*(3.0f-2.0f*tnorm);\n"
"}\n"
// NB param 'smth' (NOT 'smooth') deliberately avoids any chance of clashing with a reserved/qualifier
// word in a strict OpenCL-C compiler — a reserved var name = clBuildProgram FAIL = all rendering dead.
"__kernel void k_chroma(__global float* over,float kr,float kg,float kb,float sim,float smth){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float pr=over[i+0], pg=over[i+1], pb=over[i+2];\n"
"  float plum=pr*0.299f+pg*0.587f+pb*0.114f;\n"          // pixel luma (BT.601)
"  float klum=kr*0.299f+kg*0.587f+kb*0.114f;\n"          // key   luma (BT.601)
"  float dr=(pr-plum)-(kr-klum), dg=(pg-plum)-(kg-klum), db=(pb-plum)-(kb-klum);\n" // chroma-vector diff (luma removed)
"  float dist=sqrt(dr*dr+dg*dg+db*db);\n"
"  float afac=ck_smoothstep(sim, sim+smth, dist);\n"     // 0 near key -> keyed, 1 far -> opaque
"  over[i+3]=over[i+3]*afac;\n"                          // scale alpha only; rgb untouched
"}\n"
"__kernel void k_brightness(__global const float* s,__global float* d,float p){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=clamp01(s[i+0]+p); d[i+1]=clamp01(s[i+1]+p); d[i+2]=clamp01(s[i+2]+p); d[i+3]=s[i+3];\n"
"}\n"
"__kernel void k_contrast(__global const float* s,__global float* d,float p){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=clamp01((s[i+0]-0.5f)*p+0.5f); d[i+1]=clamp01((s[i+1]-0.5f)*p+0.5f); d[i+2]=clamp01((s[i+2]-0.5f)*p+0.5f); d[i+3]=s[i+3];\n"
"}\n"
// saturation about luma (BT.601), in-place per-pixel: s=0 -> grayscale, s=1 -> identity
"__kernel void k_saturation(__global float* d,float sat){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float luma=d[i+0]*0.299f+d[i+1]*0.587f+d[i+2]*0.114f;\n"
"  d[i+0]=clamp01(luma+(d[i+0]-luma)*sat); d[i+1]=clamp01(luma+(d[i+1]-luma)*sat); d[i+2]=clamp01(luma+(d[i+2]-luma)*sat);\n"
"}\n"
// 3D-LUT trilinear, RED-fastest: idx=((b*N+g)*N+r)*3 ; mixed by amt
"__kernel void k_lut3d(__global const float* s,__global float* d,__global const float* lut,int n,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(s[i+0]), g=clamp01(s[i+1]), b=clamp01(s[i+2]); float fn1=(float)(n-1);\n"
"  float fr=r*fn1, fg=g*fn1, fb=b*fn1; int r0=(int)fr,g0=(int)fg,b0=(int)fb;\n"
"  int r1=r0+1,g1=g0+1,b1=b0+1; if(r1>n-1)r1=n-1; if(g1>n-1)g1=n-1; if(b1>n-1)b1=n-1;\n"
"  float dr=fr-(float)r0, dg=fg-(float)g0, db=fb-(float)b0;\n"
"  int i000=((b0*n+g0)*n+r0)*3,i100=((b0*n+g0)*n+r1)*3,i010=((b0*n+g1)*n+r0)*3,i110=((b0*n+g1)*n+r1)*3;\n"
"  int i001=((b1*n+g0)*n+r0)*3,i101=((b1*n+g0)*n+r1)*3,i011=((b1*n+g1)*n+r0)*3,i111=((b1*n+g1)*n+r1)*3;\n"
"  float w000=(1.0f-dr)*(1.0f-dg)*(1.0f-db),w100=dr*(1.0f-dg)*(1.0f-db),w010=(1.0f-dr)*dg*(1.0f-db),w110=dr*dg*(1.0f-db);\n"
"  float w001=(1.0f-dr)*(1.0f-dg)*db,w101=dr*(1.0f-dg)*db,w011=(1.0f-dr)*dg*db,w111=dr*dg*db;\n"
"  for(int c=0;c<3;c++){\n"
"    float v=lut[i000+c]*w000+lut[i100+c]*w100+lut[i010+c]*w010+lut[i110+c]*w110+lut[i001+c]*w001+lut[i101+c]*w101+lut[i011+c]*w011+lut[i111+c]*w111;\n"
"    float orig=s[i+c]; d[i+c]=clamp01(orig+(v-orig)*amt);\n"
"  }\n"
"  d[i+3]=s[i+3];\n"
"}\n"
// procedural VHS (sin-hash noise — approximate parity vs MAX, sin backend differs)
"__kernel void k_vhs(__global const float* s,__global float* d,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int sh=(int)(1.0f+amt*4.0f); int xr=x-sh; if(xr<0)xr=0; int xb=x+sh; if(xb>VW-1)xb=VW-1;\n"
"  float rr=s[IDX(xr,y)+0], gg=s[i+1], bb=s[IDX(xb,y)+2];\n"
"  float sl=1.0f; if((y%2)==0) sl=1.0f-0.20f*amt;\n"
"  float hsh=sin((float)x*12.9898f+(float)y*78.233f)*43758.5453f; float noise=(hsh-floor(hsh)-0.5f)*0.14f*amt;\n"
"  d[i+0]=clamp01(rr*sl+noise); d[i+1]=clamp01(gg*sl+noise); d[i+2]=clamp01(bb*sl+noise); d[i+3]=s[i+3];\n"
"}\n"
// ---- transitions (t in [0,1]; alpha from base) ----
"__kernel void k_crossfade(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y); float w=t,inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+base[i+0]*inv; dst[i+1]=over[i+1]*w+base[i+1]*inv; dst[i+2]=over[i+2]*w+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_wipe_lr(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float edge=t*(float)VW; float w=(edge-(float)x)/36.0f+0.5f; if(w<0.0f)w=0.0f; if(w>1.0f)w=1.0f; float inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+base[i+0]*inv; dst[i+1]=over[i+1]*w+base[i+1]*inv; dst[i+2]=over[i+2]*w+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_wipe_rl(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float edge=t*(float)VW; float xr=(float)(VW-1)-(float)x; float w=(edge-xr)/36.0f+0.5f; if(w<0.0f)w=0.0f; if(w>1.0f)w=1.0f; float inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+base[i+0]*inv; dst[i+1]=over[i+1]*w+base[i+1]*inv; dst[i+2]=over[i+2]*w+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_wipe_up(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float edge=t*(float)VH; float yr=(float)(VH-1)-(float)y; float w=(edge-yr)/36.0f+0.5f; if(w<0.0f)w=0.0f; if(w>1.0f)w=1.0f; float inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+base[i+0]*inv; dst[i+1]=over[i+1]*w+base[i+1]*inv; dst[i+2]=over[i+2]*w+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_wipe_down(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float edge=t*(float)VH; float w=(edge-(float)y)/36.0f+0.5f; if(w<0.0f)w=0.0f; if(w>1.0f)w=1.0f; float inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+base[i+0]*inv; dst[i+1]=over[i+1]*w+base[i+1]*inv; dst[i+2]=over[i+2]*w+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_slide_lr(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float off=(1.0f-t)*(float)VW; int sx=(int)((float)x-off+0.5f);\n"
"  if(sx>=0&&sx<VW){ int si=IDX(sx,y); dst[i+0]=over[si+0]; dst[i+1]=over[si+1]; dst[i+2]=over[si+2]; }\n"
"  else { dst[i+0]=base[i+0]; dst[i+1]=base[i+1]; dst[i+2]=base[i+2]; }\n"
"  dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_zoom(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float cx=(float)VW*0.5f, cy=(float)VH*0.5f, scale=0.25f+0.75f*t;\n"
"  int sx=(int)(cx+((float)x-cx)/scale+0.5f), sy=(int)(cy+((float)y-cy)/scale+0.5f);\n"
"  if(sx>=0&&sx<VW&&sy>=0&&sy<VH){ int si=IDX(sx,sy); dst[i+0]=over[si+0]; dst[i+1]=over[si+1]; dst[i+2]=over[si+2]; }\n"
"  else { dst[i+0]=base[i+0]; dst[i+1]=base[i+1]; dst[i+2]=base[i+2]; }\n"
"  dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_dissolve(__global const float* base,__global const float* over,__global float* dst,float t,float power){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float p=power; if(p<1.0f)p=1.0f; float br=base[i+0],bg=base[i+1],bb=base[i+2];\n"
"  float luma=br*0.299f+bg*0.587f+bb*0.114f; float w=(t*(1.0f+1.0f/p)-luma)*p; if(w<0.0f)w=0.0f; if(w>1.0f)w=1.0f; float inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+br*inv; dst[i+1]=over[i+1]*w+bg*inv; dst[i+2]=over[i+2]*w+bb*inv; dst[i+3]=base[i+3];\n"
"}\n"
// ---- P2 color/transform effects (Shotcut-parity) ----
// 3-way color wheels: per channel out = clamp01( pow( clamp01(in*gain + lift), 1/gamma ) ). Identity
// at lift=0, gamma=1, gain=1 (Shotcut color filter lift_r..gain_b defaults). In-place on the OUT
// buffer; alpha untouched. White balance is FOLDED into gain by the UI (engine never sees temp/tint).
// NB: 'gar/gag/gab' = gamma per-channel, 'gnr/gng/gnb' = gain per-channel (NOT reserved words).
"__kernel void k_lgg(__global float* d,float lr,float lg,float lb,float gar,float gag,float gab,float gnr,float gng,float gnb){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float er=1.0f/(gar>1e-3f?gar:1e-3f), eg=1.0f/(gag>1e-3f?gag:1e-3f), eb=1.0f/(gab>1e-3f?gab:1e-3f);\n"
"  d[i+0]=clamp01(pow(clamp01(d[i+0]*gnr+lr),er));\n"
"  d[i+1]=clamp01(pow(clamp01(d[i+1]*gng+lg),eg));\n"
"  d[i+2]=clamp01(pow(clamp01(d[i+2]*gnb+lb),eb));\n"
"}\n"
// P5 CURVE: a 5-point master tone curve. 'ys' are the outputs at fixed inputs 0,.25,.5,.75,1; the
// input is mapped via piecewise-linear interpolation, applied to all 3 channels in place on OUTB.
// Identity (ys = 0,.25,.5,.75,1) is a no-op (caller skips). Runs AFTER blur, BEFORE look.
"__kernel void k_curve(__global float* d,float y0,float y1,float y2,float y3,float y4){\n"
"  int x=get_global_id(0),gy=get_global_id(1); if(x>=VW||gy>=VH) return; int i=IDX(x,gy);\n"
"  float ys[5]; ys[0]=y0; ys[1]=y1; ys[2]=y2; ys[3]=y3; ys[4]=y4;\n"
"  for(int c=0;c<3;c++){\n"
"    float v=clamp01(d[i+c])*4.0f; int seg=(int)v; if(seg>3)seg=3; float f=v-(float)seg;\n"
"    d[i+c]=clamp01(ys[seg]+(ys[seg+1]-ys[seg])*f);\n"
"  }\n"
"}\n"
// ---- P6 STYLIZE/UTILITY filters (Shotcut-parity). All run on the composited OUTB AFTER the curve,
// BEFORE the look, gated/skipped at their no-op defaults so an unfiltered clip is byte-identical.
// SIMPLE-FX (in place on OUTB): kind 1 invert (1-rgb), 2 sepia (BT-ish matrix), 3 grayscale (BT.601
// luma broadcast), 4 posterize (~6 quantization levels). kind 0 never reaches here (caller skips).
// 'kind' is an int, not a reserved word; alpha untouched throughout.
"__kernel void k_simplefx(__global float* d,int kind){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(d[i+0]), g=clamp01(d[i+1]), b=clamp01(d[i+2]);\n"
"  if(kind==1){ d[i+0]=1.0f-r; d[i+1]=1.0f-g; d[i+2]=1.0f-b; }\n"
"  else if(kind==2){\n"  // sepia (standard sepia matrix)
"    float sr=r*0.393f+g*0.769f+b*0.189f; float sg=r*0.349f+g*0.686f+b*0.168f; float sb=r*0.272f+g*0.534f+b*0.131f;\n"
"    d[i+0]=clamp01(sr); d[i+1]=clamp01(sg); d[i+2]=clamp01(sb);\n"
"  }\n"
"  else if(kind==3){ float lum=r*0.299f+g*0.587f+b*0.114f; d[i+0]=lum; d[i+1]=lum; d[i+2]=lum; }\n"
"  else if(kind==4){\n"  // posterize to ~6 levels per channel
"    float lev=6.0f; float lm1=lev-1.0f;\n"
"    d[i+0]=clamp01(floor(r*lm1+0.5f)/lm1); d[i+1]=clamp01(floor(g*lm1+0.5f)/lm1); d[i+2]=clamp01(floor(b*lm1+0.5f)/lm1);\n"
"  }\n"
"}\n"
// VIGNETTE (in place on OUTB): radial edge darken. 'dist' = distance of the pixel from the image
// centre normalized so a corner is ~1.0; factor=1-amt*smoothstep(0.5,0.95,dist); rgb*=factor. amt<=0
// never reaches here (caller skips). Reuses ck_smoothstep (defined above) for the soft falloff.
"__kernel void k_vignette(__global float* d,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float cx=(float)VW*0.5f, cy=(float)VH*0.5f;\n"
"  float dx=((float)x+0.5f)-cx, dy=((float)y+0.5f)-cy;\n"
"  float maxr=sqrt(cx*cx+cy*cy);\n"
"  float dist=sqrt(dx*dx+dy*dy)/(maxr>1e-4f?maxr:1e-4f);\n"
"  float factor=1.0f-amt*ck_smoothstep(0.5f,0.95f,dist); if(factor<0.0f)factor=0.0f;\n"
"  d[i+0]=clamp01(d[i+0]*factor); d[i+1]=clamp01(d[i+1]*factor); d[i+2]=clamp01(d[i+2]*factor);\n"
"}\n"
// SHARPEN (unsharp): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'. Per channel
// out=center*(1+4a) - 4*neighbour_avg*a, i.e. center*(1+4a) - a*(left+right+up+down). Edge-clamped
// neighbour sampling. amt<=0 never reaches here (caller skips). 's' is the distinct source copy so
// reads see the pre-filter pixels. 'amt'/'cen' are not reserved words.
"__kernel void k_sharpen(__global const float* s,__global float* d,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int xl=x-1; if(xl<0)xl=0; int xr=x+1; if(xr>VW-1)xr=VW-1;\n"
"  int yu=y-1; if(yu<0)yu=0; int yd=y+1; if(yd>VH-1)yd=VH-1;\n"
"  int il=IDX(xl,y), ir=IDX(xr,y), iu=IDX(x,yu), id=IDX(x,yd);\n"
"  for(int c=0;c<3;c++){\n"
"    float cen=s[i+c]; float nb=s[il+c]+s[ir+c]+s[iu+c]+s[id+c];\n"
"    d[i+c]=clamp01(cen*(1.0f+4.0f*amt) - nb*amt);\n"
"  }\n"
"  d[i+3]=s[i+3];\n"
"}\n"
// FLIP (mirror): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd' by sampling 's' at the
// flipped coordinate. mode 1 -> x mirrored (W-1-x), 2 -> y mirrored (H-1-y), 3 -> both. mode 0 never
// reaches here (caller skips). 'mode'/'sxf'/'syf' are not reserved words.
"__kernel void k_flip(__global const float* s,__global float* d,int mode){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int sxf=x, syf=y;\n"
"  if(mode==1){ sxf=VW-1-x; }\n"
"  else if(mode==2){ syf=VH-1-y; }\n"
"  else if(mode==3){ sxf=VW-1-x; syf=VH-1-y; }\n"
"  int si=IDX(sxf,syf);\n"
"  d[i+0]=s[si+0]; d[i+1]=s[si+1]; d[i+2]=s[si+2]; d[i+3]=s[si+3];\n"
"}\n"
// transform: rotate (degrees) + uniform scale about the image center, BILINEAR sampling of source 's'
// into 'd'. Inverse map: for each dst pixel, undo the scale (divide) and rotation (rotate by -rot),
// sample s. Out-of-bounds -> transparent black. Identity at rot=0, scale=1. Reads s, writes d (NEVER
// the same buffer in place — caller passes a distinct source copy). 'rot_deg'/'scl' are NOT reserved.
"__kernel void k_transform(__global const float* s,__global float* d,float rot_deg,float scl,int w,int h){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int di=IDX(x,y);\n"
"  float cx=(float)w*0.5f, cy=(float)h*0.5f;\n"
"  float sf=(scl>1e-4f?scl:1e-4f);\n"
"  float rad=-rot_deg*0.01745329252f; float cs=cos(rad), sn=sin(rad);\n"   // inverse rotation: -rot
"  float dx=((float)x+0.5f)-cx, dy=((float)y+0.5f)-cy;\n"
"  float rx=(dx*cs - dy*sn)/sf, ry=(dx*sn + dy*cs)/sf;\n"                   // inverse: rotate then /scale
"  float sx=rx+cx-0.5f, sy=ry+cy-0.5f;\n"
"  int x0=(int)floor(sx), y0=(int)floor(sy); int x1=x0+1, y1=y0+1;\n"
"  float fx=sx-(float)x0, fy=sy-(float)y0;\n"
"  if(x1<0||y1<0||x0>w-1||y0>h-1){ d[di+0]=0.0f; d[di+1]=0.0f; d[di+2]=0.0f; d[di+3]=0.0f; return; }\n"
"  int cx0=x0<0?0:(x0>w-1?w-1:x0), cx1=x1<0?0:(x1>w-1?w-1:x1);\n"
"  int cy0=y0<0?0:(y0>h-1?h-1:y0), cy1=y1<0?0:(y1>h-1?h-1:y1);\n"
"  int i00=IDX(cx0,cy0), i10=IDX(cx1,cy0), i01=IDX(cx0,cy1), i11=IDX(cx1,cy1);\n"
"  float w00=(1.0f-fx)*(1.0f-fy), w10=fx*(1.0f-fy), w01=(1.0f-fx)*fy, w11=fx*fy;\n"
"  for(int c=0;c<4;c++){ d[di+c]=s[i00+c]*w00+s[i10+c]*w10+s[i01+c]*w01+s[i11+c]*w11; }\n"
"}\n"
// separable gaussian blur, horizontal pass: read s, write d. 'sigma'<=0 => copy. radius=ceil(2*sigma)
// capped at 32; weights exp(-x^2/(2 sigma^2)) normalized. Edge clamps the sample coordinate. Vertical
// pass below mirrors it on the y-axis. 'sig'/'rad'/'wsum' are NOT reserved words.
"__kernel void k_blur_h(__global const float* s,__global float* d,float sigma){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  if(sigma<=0.0f){ d[i+0]=s[i+0]; d[i+1]=s[i+1]; d[i+2]=s[i+2]; d[i+3]=s[i+3]; return; }\n"
"  int rad=(int)ceil(2.0f*sigma); if(rad>32)rad=32; if(rad<1)rad=1;\n"
"  float inv2s2=1.0f/(2.0f*sigma*sigma);\n"
"  float acc0=0.0f,acc1=0.0f,acc2=0.0f,wsum=0.0f;\n"
"  for(int k=-rad;k<=rad;k++){\n"
"    int xx=x+k; if(xx<0)xx=0; if(xx>VW-1)xx=VW-1; int si=IDX(xx,y);\n"
"    float wgt=exp(-(float)(k*k)*inv2s2);\n"
"    acc0+=s[si+0]*wgt; acc1+=s[si+1]*wgt; acc2+=s[si+2]*wgt; wsum+=wgt;\n"
"  }\n"
"  float invw=1.0f/wsum; d[i+0]=acc0*invw; d[i+1]=acc1*invw; d[i+2]=acc2*invw; d[i+3]=s[i+3];\n"
"}\n"
"__kernel void k_blur_v(__global const float* s,__global float* d,float sigma){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  if(sigma<=0.0f){ d[i+0]=s[i+0]; d[i+1]=s[i+1]; d[i+2]=s[i+2]; d[i+3]=s[i+3]; return; }\n"
"  int rad=(int)ceil(2.0f*sigma); if(rad>32)rad=32; if(rad<1)rad=1;\n"
"  float inv2s2=1.0f/(2.0f*sigma*sigma);\n"
"  float acc0=0.0f,acc1=0.0f,acc2=0.0f,wsum=0.0f;\n"
"  for(int k=-rad;k<=rad;k++){\n"
"    int yy=y+k; if(yy<0)yy=0; if(yy>VH-1)yy=VH-1; int si=IDX(x,yy);\n"
"    float wgt=exp(-(float)(k*k)*inv2s2);\n"
"    acc0+=s[si+0]*wgt; acc1+=s[si+1]*wgt; acc2+=s[si+2]*wgt; wsum+=wgt;\n"
"  }\n"
"  float invw=1.0f/wsum; d[i+0]=acc0*invw; d[i+1]=acc1*invw; d[i+2]=acc2*invw; d[i+3]=s[i+3];\n"
"}\n"
// ---- P7 COLOR filters (Shotcut-parity). Both run on the composited OUTB AFTER the P6 filters (flip),
// BEFORE the look, in place per-pixel (own pixel only), in the pinned order HSL -> LEVELS. Each is a
// no-op at its identity default (HSL hue0/sat1/light0 ; LEVELS inb0/inw1/gam1) so the caller skips it
// and an unfiltered clip is byte-identical. NB: all var names avoid reserved OpenCL words (no
// local/global/half/double/kernel/constant/uniform/...); reserved-word var = clBuildProgram FAIL.
// HSL ADJUST: RGB->HSL (standard hexcone), then hue += hue_deg (wrapped mod 360), saturation *= sat,
// lightness += light, then HSL->RGB and clamp01. The hue/sat/lightness are the usual definitions:
//   maxc=max(r,g,b), minc=min(r,g,b), chr=maxc-minc ; lightness L=(maxc+minc)/2 ;
//   saturation S = chr / (1 - |2L-1|) (0 when chr==0) ; hue H from which channel is max (degrees).
// Reconstruction uses chr2 = (1-|2L'-1|)*S' and a per-channel hue ramp, matching the inverse.
"__kernel void k_hsl(__global float* d,float hue_deg,float sat,float light){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(d[i+0]), g=clamp01(d[i+1]), b=clamp01(d[i+2]);\n"
"  float maxc=r; if(g>maxc)maxc=g; if(b>maxc)maxc=b;\n"
"  float minc=r; if(g<minc)minc=g; if(b<minc)minc=b;\n"
"  float chr=maxc-minc;\n"
"  float lgt=(maxc+minc)*0.5f;\n"
"  float hue=0.0f;\n"
"  if(chr>1e-6f){\n"
"    if(maxc==r){ hue=fmod((g-b)/chr,6.0f); }\n"
"    else if(maxc==g){ hue=(b-r)/chr+2.0f; }\n"
"    else { hue=(r-g)/chr+4.0f; }\n"
"    hue*=60.0f; if(hue<0.0f) hue+=360.0f;\n"
"  }\n"
"  float satr=0.0f;\n"
"  float denom=1.0f-fabs(2.0f*lgt-1.0f);\n"
"  if(denom>1e-6f) satr=chr/denom;\n"
"  hue+=hue_deg; hue=fmod(hue,360.0f); if(hue<0.0f) hue+=360.0f;\n"
"  satr*=sat; if(satr<0.0f) satr=0.0f; if(satr>1.0f) satr=1.0f;\n"
"  lgt+=light; if(lgt<0.0f) lgt=0.0f; if(lgt>1.0f) lgt=1.0f;\n"
"  float chr2=(1.0f-fabs(2.0f*lgt-1.0f))*satr;\n"
"  float hp=hue/60.0f;\n"
"  float xc=chr2*(1.0f-fabs(fmod(hp,2.0f)-1.0f));\n"
"  float mm=lgt-chr2*0.5f;\n"
"  float rr=0.0f,gg=0.0f,bb=0.0f;\n"
"  if(hp<1.0f){ rr=chr2; gg=xc; bb=0.0f; }\n"
"  else if(hp<2.0f){ rr=xc; gg=chr2; bb=0.0f; }\n"
"  else if(hp<3.0f){ rr=0.0f; gg=chr2; bb=xc; }\n"
"  else if(hp<4.0f){ rr=0.0f; gg=xc; bb=chr2; }\n"
"  else if(hp<5.0f){ rr=xc; gg=0.0f; bb=chr2; }\n"
"  else { rr=chr2; gg=0.0f; bb=xc; }\n"
"  d[i+0]=clamp01(rr+mm); d[i+1]=clamp01(gg+mm); d[i+2]=clamp01(bb+mm);\n"
"}\n"
// LEVELS: per channel out = clamp01( pow( clamp01((c-in_black)/max(in_white-in_black,1e-3)),
// 1/max(gamma,1e-3) ) ). Identity in_black=0,in_white=1,gamma=1 => caller skips. 'inb'/'inw'/'gam'
// are NOT reserved words.
"__kernel void k_levels(__global float* d,float in_black,float in_white,float gamma){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float spanv=in_white-in_black; if(spanv<1e-3f) spanv=1e-3f;\n"
"  float ginv=gamma; if(ginv<1e-3f) ginv=1e-3f; ginv=1.0f/ginv;\n"
"  for(int c=0;c<3;c++){\n"
"    float v=clamp01((d[i+c]-in_black)/spanv);\n"
"    d[i+c]=clamp01(pow(v,ginv));\n"
"  }\n"
"}\n"
// ---- P8 STYLIZE-2 filters (Shotcut-parity). Both run on the composited OUTB AFTER the P7 color
// filters (levels), BEFORE the look, in the pinned order MOSAIC -> GRADIENT-MAP. Each is a no-op
// at its default (mosaic block<=1 ; gmap amt<=0) so the caller skips it and an unfiltered clip is
// byte-identical. NB: all var names avoid reserved OpenCL words (no local/global/half/double/
// kernel/constant/uniform/...); 'block' is NOT a reserved word and matches the pinned API.
// MOSAIC (pixelate): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'. For each pixel,
// snap to the block top-left (bx=(x/block)*block, by=(y/block)*block) and copy that single source
// pixel across the whole block — so a block reads ONE source pixel (from the distinct g_tmp copy,
// avoiding the read/write race an in-place mosaic would hit). block<=1 never reaches here (caller
// skips). 'block'/'bx'/'by' are not reserved words.
"__kernel void k_mosaic(__global const float* s,__global float* d,int block){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int bx=(x/block)*block, by=(y/block)*block;\n"
"  if(bx>VW-1)bx=VW-1; if(by>VH-1)by=VH-1; int si=IDX(bx,by);\n"
"  d[i+0]=s[si+0]; d[i+1]=s[si+1]; d[i+2]=s[si+2]; d[i+3]=s[si+3];\n"
"}\n"
// GRADIENT MAP: luma -> colour ramp, IN PLACE on OUTB (per-own-pixel, no scratch). luma=dot(rgb,
// [.299,.587,.114]); mapped=mix(lo,hi,luma) per channel; rgb=mix(rgb,mapped,amt). amt<=0 never
// reaches here (caller skips). 'lr/lg/lb' = shadow colour, 'hr/hg/hb' = highlight colour, 'amt' =
// blend; 'mr/mg/mb' = mapped colour. None are reserved words. Alpha untouched.
"__kernel void k_gmap(__global float* d,float amt,float lr,float lg,float lb,float hr,float hg,float hb){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(d[i+0]), g=clamp01(d[i+1]), b=clamp01(d[i+2]);\n"
"  float luma=r*0.299f+g*0.587f+b*0.114f;\n"
"  float mr=lr+(hr-lr)*luma, mg=lg+(hg-lg)*luma, mb=lb+(hb-lb)*luma;\n"
"  d[i+0]=clamp01(r+(mr-r)*amt); d[i+1]=clamp01(g+(mg-g)*amt); d[i+2]=clamp01(b+(mb-b)*amt);\n"
"}\n"
// ---- P9 FX filters (Shotcut-parity). All run on the composited OUTB AFTER the P8 gradient-map,
// BEFORE the look, in the pinned order DENOISE -> GLOW -> RGB-SHIFT. Each is a no-op at its
// default (denoise<=0 ; glow amt<=0 ; rgbshift off<=0) so the caller skips it and an unfiltered
// clip is byte-identical. NB: all var names avoid reserved OpenCL words (no local/global/half/
// double/kernel/constant/uniform/...); a reserved-word var = clBuildProgram FAIL = all rendering
// dead. The reserved-word avoidance below is deliberate (e.g. 'srng'/'spat'/'wgt'/'strength').
// DENOISE: edge-preserving 5x5 BILATERAL smooth. reads source 's' (a copy of OUTB in g_tmp),
// writes OUTB 'd'. For each output pixel, sum the 5x5 neighbourhood weighted by a SPATIAL gaussian
// (distance falloff) times a COLOUR-RANGE gaussian (penalize neighbours whose colour differs from
// the centre — this is what preserves edges). The weighted average is then blended back toward the
// centre by `strength` (0 = identity, 1 = full bilateral). strength<=0 never reaches here (caller
// skips). 'strength'/'srng'/'spat'/'wgt'/'cen*'/'acc*' are NOT reserved words.
"__kernel void k_denoise(__global const float* s,__global float* d,float strength){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float cenr=s[i+0], ceng=s[i+1], cenb=s[i+2];\n"
"  float spat2=2.0f*1.6f*1.6f;\n"        // spatial gaussian variance term (sigma~1.6 over a 5x5)
"  float srng2=2.0f*0.12f*0.12f;\n"      // colour-range gaussian variance term (sigma~0.12)
"  float accr=0.0f,accg=0.0f,accb=0.0f,wsum=0.0f;\n"
"  for(int ky=-2;ky<=2;ky++){\n"
"    int yy=y+ky; if(yy<0)yy=0; if(yy>VH-1)yy=VH-1;\n"
"    for(int kx=-2;kx<=2;kx++){\n"
"      int xx=x+kx; if(xx<0)xx=0; if(xx>VW-1)xx=VW-1; int si=IDX(xx,yy);\n"
"      float nr=s[si+0], ng=s[si+1], nb=s[si+2];\n"
"      float spat=exp(-(float)(kx*kx+ky*ky)/spat2);\n"
"      float dcr=nr-cenr, dcg=ng-ceng, dcb=nb-cenb;\n"
"      float srng=exp(-(dcr*dcr+dcg*dcg+dcb*dcb)/srng2);\n"
"      float wgt=spat*srng;\n"
"      accr+=nr*wgt; accg+=ng*wgt; accb+=nb*wgt; wsum+=wgt;\n"
"    }\n"
"  }\n"
"  float invw=(wsum>1e-6f)?1.0f/wsum:0.0f;\n"
"  float fr=accr*invw, fg=accg*invw, fb=accb*invw;\n"
"  float sclamp=strength; if(sclamp<0.0f)sclamp=0.0f; if(sclamp>1.0f)sclamp=1.0f;\n"
"  d[i+0]=clamp01(cenr+(fr-cenr)*sclamp);\n"
"  d[i+1]=clamp01(ceng+(fg-ceng)*sclamp);\n"
"  d[i+2]=clamp01(cenb+(fb-cenb)*sclamp);\n"
"  d[i+3]=s[i+3];\n"
"}\n"
// GLOW step 1 — BRIGHT-PASS extract: read source 's' (OUTB), write 'd' (g_tmp2). Where the pixel's
// luma exceeds `thr` keep its rgb, else 0. Alpha set to 1 (the extract buffer is blurred then added
// back, so its alpha is irrelevant). 'thr'/'lum' are NOT reserved words.
"__kernel void k_glow_extract(__global const float* s,__global float* d,float thr){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(s[i+0]), g=clamp01(s[i+1]), b=clamp01(s[i+2]);\n"
"  float lum=r*0.299f+g*0.587f+b*0.114f;\n"
"  if(lum>thr){ d[i+0]=r; d[i+1]=g; d[i+2]=b; }\n"
"  else { d[i+0]=0.0f; d[i+1]=0.0f; d[i+2]=0.0f; }\n"
"  d[i+3]=1.0f;\n"
"}\n"
// GLOW step 3 — COMBINE: OUTB 'd' += amt * blurred-bright-pass 'g' (the buffer g_glow holds the
// blurred bright pass). Clamped to [0,1]. This brightens dark pixels AROUND a bright region (bloom
// halo). 'amt' is NOT a reserved word.
"__kernel void k_glow_combine(__global float* d,__global const float* g,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=clamp01(d[i+0]+amt*g[i+0]); d[i+1]=clamp01(d[i+1]+amt*g[i+1]); d[i+2]=clamp01(d[i+2]+amt*g[i+2]);\n"
"}\n"
// RGB-SHIFT (chromatic aberration): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'.
// out.r samples s.r at (x+off,y), out.b samples s.b at (x-off,y), out.g/out.a sample s at (x,y).
// Sample x-coords are clamped to [0,VW-1]. off<=0 never reaches here (caller skips). 'off'/'xr'/'xb'
// are NOT reserved words.
"__kernel void k_rgbshift(__global const float* s,__global float* d,int off){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int xr=x+off; if(xr<0)xr=0; if(xr>VW-1)xr=VW-1;\n"
"  int xb=x-off; if(xb<0)xb=0; if(xb>VW-1)xb=VW-1;\n"
"  d[i+0]=s[IDX(xr,y)+0]; d[i+1]=s[i+1]; d[i+2]=s[IDX(xb,y)+2]; d[i+3]=s[i+3];\n"
"}\n";

// cached kernel objects (clCreateKernel once)
static cl_kernel K(const char* name){ cl_int e; cl_kernel k = clCreateKernel(g_prog, name, &e); return (e==CL_SUCCESS)?k:NULL; }
static cl_kernel kUnpack,kPack,kCopy,kComposite,kPip,kBright,kContrast,kSat,kLut,kVhs;
static cl_kernel kHistClear,kHist;
static cl_kernel kGridClear,kWaveAcc,kWaveImg,kVecAcc,kVecImg;
static cl_kernel kParadeClear,kParadeAcc,kParadeImg;
static cl_kernel kLgg,kTransform,kBlurH,kBlurV; // P2 color/transform effects
static cl_kernel kCurve; // P5 master tone curve
static cl_kernel kSimplefx,kVignette,kSharpen,kFlip; // P6 stylize/utility filters
static cl_kernel kHsl,kLevels; // P7 color filters (HSL adjust + levels)
static cl_kernel kMosaic,kGmap; // P8 stylize-2 filters (mosaic pixelate + gradient map)
static cl_kernel kDenoise,kGlowExtract,kGlowCombine,kRgbshift; // P9 fx filters (denoise/glow/rgb-shift)
static cl_kernel kChroma; // P4 chroma key (green-screen) on the OVER buffer
static cl_kernel kTrans[8]; // 0..7

static int launch(cl_kernel k){
  size_t gl[2] = { GVW, GVH };
  size_t lo[2] = { 8, 8 };
  cl_int e = clEnqueueNDRangeKernel(g_q, k, 2, NULL, gl, lo, 0, NULL, NULL);
  return e==CL_SUCCESS ? 0 : -1;
}

int fpx_gpu_init(void){
  if (g_ready) return 0;
  cl_uint np=0; cl_platform_id plat[8];
  if (clGetPlatformIDs(8, plat, &np)!=CL_SUCCESS || np==0) return -1;
  cl_int e;
  // first platform with a GPU device
  for (cl_uint p=0;p<np && !g_dev;p++){
    cl_uint nd=0; if (clGetDeviceIDs(plat[p], CL_DEVICE_TYPE_GPU, 1, &g_dev, &nd)!=CL_SUCCESS || nd==0) g_dev=NULL;
  }
  if (!g_dev){ // fall back to any device
    for (cl_uint p=0;p<np && !g_dev;p++){ cl_uint nd=0; clGetDeviceIDs(plat[p], CL_DEVICE_TYPE_ALL, 1, &g_dev, &nd); }
  }
  if (!g_dev) return -2;
  g_ctx = clCreateContext(NULL, 1, &g_dev, NULL, NULL, &e); if (e!=CL_SUCCESS) return -3;
  g_q = clCreateCommandQueue(g_ctx, g_dev, 0, &e); if (e!=CL_SUCCESS) return -4;
  g_prog = clCreateProgramWithSource(g_ctx, 1, &KSRC, NULL, &e); if (e!=CL_SUCCESS) return -5;
  char opt[64]; snprintf(opt, sizeof opt, "-DVW=%d -DVH=%d", GVW, GVH);
  e = clBuildProgram(g_prog, 1, &g_dev, opt, NULL, NULL);
  if (e!=CL_SUCCESS){
    size_t ls=0; clGetProgramBuildInfo(g_prog, g_dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ls);
    char* log=(char*)malloc(ls+1); clGetProgramBuildInfo(g_prog, g_dev, CL_PROGRAM_BUILD_LOG, ls, log, NULL); log[ls]=0;
    fprintf(stderr, "fpx_gpu clBuildProgram failed:\n%s\n", log); free(log); return -6;
  }
  for (int i=0;i<NBUF;i++){ g_buf[i]=clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, GN*sizeof(float), NULL, &e); if(e!=CL_SUCCESS) return -7; }
  g_lut   = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY, MAXLUTF*sizeof(float), NULL, &e); if(e!=CL_SUCCESS) return -8;
  // READ_WRITE (not READ_ONLY): g_stage is used BOTH directions — host writes + k_unpack reads it on
  // upload, AND k_pack WRITES it on download_u8 (then host reads). A READ_ONLY buffer written by a
  // kernel is undefined behaviour, so the correct flag is READ_WRITE. (The primary cause of the
  // N-layer fold's black band was the non-blocking upload below; this is a related correctness fix.)
  g_stage = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, GN*sizeof(unsigned char), NULL, &e); if(e!=CL_SUCCESS) return -9;
  g_hist  = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, 768*sizeof(int), NULL, &e); if(e!=CL_SUCCESS) return -11;
  g_grid  = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, 65536*sizeof(int), NULL, &e); if(e!=CL_SUCCESS) return -12;
  g_scope = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, 256*256*4, NULL, &e); if(e!=CL_SUCCESS) return -13;
  g_parade= clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, 3*65536*sizeof(int), NULL, &e); if(e!=CL_SUCCESS) return -15;
  g_tmp   = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, GN*sizeof(float), NULL, &e); if(e!=CL_SUCCESS) return -17; // P2 scratch
  g_tmp2  = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, GN*sizeof(float), NULL, &e); if(e!=CL_SUCCESS) return -30; // P9 scratch #2 (glow)
  kUnpack=K("k_unpack"); kPack=K("k_pack"); kCopy=K("k_copy"); kComposite=K("k_composite"); kPip=K("k_pip");
  kBright=K("k_brightness"); kContrast=K("k_contrast"); kSat=K("k_saturation"); kLut=K("k_lut3d"); kVhs=K("k_vhs");
  kHistClear=K("k_hist_clear"); kHist=K("k_hist");
  kGridClear=K("k_grid_clear"); kWaveAcc=K("k_wave_acc"); kWaveImg=K("k_wave_img"); kVecAcc=K("k_vec_acc"); kVecImg=K("k_vec_img");
  kParadeClear=K("k_parade_clear"); kParadeAcc=K("k_parade_acc"); kParadeImg=K("k_parade_img");
  kLgg=K("k_lgg"); kTransform=K("k_transform"); kBlurH=K("k_blur_h"); kBlurV=K("k_blur_v"); // P2
  kChroma=K("k_chroma"); // P4 chroma key
  kCurve=K("k_curve");   // P5 master tone curve
  kSimplefx=K("k_simplefx"); kVignette=K("k_vignette"); kSharpen=K("k_sharpen"); kFlip=K("k_flip"); // P6
  kHsl=K("k_hsl"); kLevels=K("k_levels"); // P7 color filters
  kMosaic=K("k_mosaic"); kGmap=K("k_gmap"); // P8 stylize-2 filters
  kDenoise=K("k_denoise"); kGlowExtract=K("k_glow_extract"); kGlowCombine=K("k_glow_combine"); kRgbshift=K("k_rgbshift"); // P9 fx
  kTrans[0]=K("k_crossfade"); kTrans[1]=K("k_wipe_lr"); kTrans[2]=K("k_wipe_rl"); kTrans[3]=K("k_wipe_up");
  kTrans[4]=K("k_wipe_down"); kTrans[5]=K("k_slide_lr"); kTrans[6]=K("k_zoom"); kTrans[7]=K("k_dissolve");
  if(!kUnpack||!kPack||!kComposite||!kPip||!kBright||!kContrast||!kLut||!kVhs||!kTrans[7]) return -10;
  // scope kernels must build too — else a SCOPE command later launches a NULL kernel (segfault).
  if(!kHistClear||!kHist||!kGridClear||!kWaveAcc||!kWaveImg||!kVecAcc||!kVecImg) return -14;
  // RGB parade kernels (Triad-B P1) — same NULL-kernel guard so a SCOPE 3 never segfaults.
  if(!kParadeClear||!kParadeAcc||!kParadeImg) return -16;
  // P2 color/transform kernels — same NULL-kernel guard so a compose that runs them never segfaults.
  if(!kLgg||!kTransform||!kBlurH||!kBlurV) return -18;
  // P4 chroma-key kernel — same NULL-kernel guard so a compose that runs it never segfaults.
  if(!kChroma) return -19;
  if(!kCurve) return -20;
  // P6 stylize/utility kernels — same NULL-kernel guard so a compose that runs them never segfaults.
  if(!kSimplefx) return -21;
  if(!kVignette) return -22;
  if(!kSharpen) return -23;
  if(!kFlip) return -24;
  // k_copy backs transform/blur (P2) AND now sharpen/flip (P6): if it failed to create, those
  // wrappers would launch a NULL kernel and segfault while fpx_gpu_init falsely reported ready.
  // (Pre-existing gap; folded in here since the P6 filters newly depend on it.)
  if(!kCopy) return -25;
  // P7 color kernels (HSL adjust + levels) — same NULL-kernel guard so a compose that runs them
  // (after the P6 flip, before the look) never launches a NULL kernel / segfaults.
  if(!kHsl) return -26;
  if(!kLevels) return -27;
  // P8 stylize-2 kernels (mosaic pixelate + gradient map) — same NULL-kernel guard so a compose that
  // runs them (after the P7 levels, before the look) never launches a NULL kernel / segfaults.
  if(!kMosaic) return -28;
  if(!kGmap) return -29;
  // P9 fx kernels (denoise bilateral / glow bright-pass+combine / rgb-shift) — same NULL-kernel guard
  // so a compose that runs them (after the P8 gradient-map, before the look) never launches a NULL
  // kernel / segfaults. g_tmp2 (the glow scratch #2) gets its own alloc guard (-30) above.
  if(!kDenoise) return -31;
  if(!kGlowExtract) return -32;
  if(!kGlowCombine) return -33;
  if(!kRgbshift) return -34;
  g_ready=1; return 0;
}

// upload an RGBA8 frame to slot (0=base,1=over,2=trans): staging u8 -> unpack -> float buf.
// The write is BLOCKING (CL_TRUE): the host `rgba8` pointer is a transient caller buffer (a decoded
// Rust Vec that is dropped/reused right after this returns). A non-blocking (CL_FALSE) write lets the
// host->device DMA outlive the call and read freed memory → a corrupt, NON-DETERMINISTIC frame
// (manifested as a black band over the lower part of the image, worst on the cold first compose and
// in the back-to-back N-layer render fold). CL_TRUE guarantees the buffer is fully consumed first.
int fpx_gpu_upload_u8(int slot, const unsigned char* rgba8){
  if(!g_ready||slot<0||slot>2||!rgba8) return -1;
  if(clEnqueueWriteBuffer(g_q,g_stage,CL_TRUE,0,GN*sizeof(unsigned char),rgba8,0,NULL,NULL)!=CL_SUCCESS) return -2;
  clSetKernelArg(kUnpack,0,sizeof(cl_mem),&g_stage); clSetKernelArg(kUnpack,1,sizeof(cl_mem),&g_buf[slot]);
  return launch(kUnpack);
}
int fpx_gpu_upload_f32(int slot, const float* f32){
  if(!g_ready||slot<0||slot>2||!f32) return -1; // CL_TRUE: transient host buffer (see fpx_gpu_upload_u8).
  return clEnqueueWriteBuffer(g_q,g_buf[slot],CL_TRUE,0,GN*sizeof(float),f32,0,NULL,NULL)==CL_SUCCESS?0:-2;
}
int fpx_gpu_upload_lut(const float* lut, int nfloats){
  if(!g_ready||!lut||nfloats<=0||nfloats>MAXLUTF) return -1; // CL_TRUE: transient host buffer.
  return clEnqueueWriteBuffer(g_q,g_lut,CL_TRUE,0,nfloats*sizeof(float),lut,0,NULL,NULL)==CL_SUCCESS?0:-2;
}

// track1 = transition(base,trans,t,param)  OR (tt<0) composite(base,trans,0)=base
void fpx_gpu_track1(int tt, float t, float param){
  if(!g_ready) return;
  if(tt<0||tt>7){ clSetKernelArg(kComposite,0,sizeof(cl_mem),&g_buf[BASE]); clSetKernelArg(kComposite,1,sizeof(cl_mem),&g_buf[TRANS]);
    clSetKernelArg(kComposite,2,sizeof(cl_mem),&g_buf[TRACK1]); float z=0.0f; clSetKernelArg(kComposite,3,sizeof(float),&z); launch(kComposite); return; }
  cl_kernel k=kTrans[tt];
  clSetKernelArg(k,0,sizeof(cl_mem),&g_buf[BASE]); clSetKernelArg(k,1,sizeof(cl_mem),&g_buf[TRANS]); clSetKernelArg(k,2,sizeof(cl_mem),&g_buf[TRACK1]);
  clSetKernelArg(k,3,sizeof(float),&t);
  if(tt==7) clSetKernelArg(k,4,sizeof(float),&param);
  launch(k);
}
// P2 TRANSFORM: rotate (degrees) + uniform scale the BASE frame (TRACK1 buffer) about its center,
// bilinear. Runs RIGHT AFTER fpx_gpu_track1, BEFORE pip — so the PiP overlay composites onto the
// already-transformed base. Identity at rot=0,scale=1 (skipped → TRACK1 untouched, zero cost). Since
// the kernel cannot read+write the same buffer safely, copy TRACK1→g_tmp first, then transform the
// COPY back into TRACK1.
void fpx_gpu_transform(float rot_deg, float scale){
  if(!g_ready) return;
  int identity = (rot_deg>-0.001f && rot_deg<0.001f) && (scale>0.999f && scale<1.001f);
  if(identity) return; // identity transform: leave TRACK1 as-is.
  // TRACK1 -> g_tmp (source copy), then transform g_tmp -> TRACK1.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[TRACK1]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  int w=GVW, h=GVH;
  clSetKernelArg(kTransform,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kTransform,1,sizeof(cl_mem),&g_buf[TRACK1]);
  clSetKernelArg(kTransform,2,sizeof(float),&rot_deg); clSetKernelArg(kTransform,3,sizeof(float),&scale);
  clSetKernelArg(kTransform,4,sizeof(int),&w); clSetKernelArg(kTransform,5,sizeof(int),&h);
  launch(kTransform);
}
// P4 CHROMA KEY: zero/soften the OVER buffer's ALPHA where the pixel matches the key colour, IN PLACE
// on the OVER buffer. Runs AFTER the over upload (+ any over transform) and BEFORE fpx_gpu_pip, so the
// pip composite (`over.a*op`) shows the base through the keyed pixels. The caller only invokes this
// when the clip's chroma is ENABLED (ck_on==1); a disabled clip never calls it, so the OVER alpha is
// untouched and the composite is byte-identical to P3. RGB is never modified.
void fpx_gpu_chroma(float kr, float kg, float kb, float sim, float smooth){
  if(!g_ready) return;
  clSetKernelArg(kChroma,0,sizeof(cl_mem),&g_buf[OVER]);
  clSetKernelArg(kChroma,1,sizeof(float),&kr); clSetKernelArg(kChroma,2,sizeof(float),&kg); clSetKernelArg(kChroma,3,sizeof(float),&kb);
  clSetKernelArg(kChroma,4,sizeof(float),&sim); clSetKernelArg(kChroma,5,sizeof(float),&smooth);
  launch(kChroma);
}
// in = composite_pip(track1, over, op, px,py,pw,ph)
void fpx_gpu_pip(float op, float px, float py, float pw, float ph){
  if(!g_ready) return;
  clSetKernelArg(kPip,0,sizeof(cl_mem),&g_buf[TRACK1]); clSetKernelArg(kPip,1,sizeof(cl_mem),&g_buf[OVER]); clSetKernelArg(kPip,2,sizeof(cl_mem),&g_buf[INB]);
  clSetKernelArg(kPip,3,sizeof(float),&op); clSetKernelArg(kPip,4,sizeof(float),&px); clSetKernelArg(kPip,5,sizeof(float),&py);
  clSetKernelArg(kPip,6,sizeof(float),&pw); clSetKernelArg(kPip,7,sizeof(float),&ph); launch(kPip);
}
// mid = brightness(in); out = contrast(mid); out = saturation(out) in-place (skip if sat==1)
void fpx_gpu_grade(float bright, float contrast, float sat){
  if(!g_ready) return;
  clSetKernelArg(kBright,0,sizeof(cl_mem),&g_buf[INB]); clSetKernelArg(kBright,1,sizeof(cl_mem),&g_buf[MID]); clSetKernelArg(kBright,2,sizeof(float),&bright); launch(kBright);
  clSetKernelArg(kContrast,0,sizeof(cl_mem),&g_buf[MID]); clSetKernelArg(kContrast,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kContrast,2,sizeof(float),&contrast); launch(kContrast);
  if(sat<0.999f || sat>1.001f){ clSetKernelArg(kSat,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kSat,1,sizeof(float),&sat); launch(kSat); }
}
// PER-CLIP grade (Triad-B P1): grade the PiP-composite buffer (INB) IN PLACE, BEFORE the program
// grade. Same kernels/semantics as fpx_gpu_grade, but routes INB→MID(bright)→INB(contrast)→INB(sat)
// so the result lands back in INB — then the caller runs the program fpx_gpu_grade(INB→…→OUTB) on
// top, giving the documented "per-clip first, then program" additive order. A neutral grade
// (bright 0, contrast 1, sat 1) is a no-op (skipped) so a clip with the default grade costs nothing.
void fpx_gpu_grade_clip(float bright, float contrast, float sat){
  if(!g_ready) return;
  int neutral = (bright>-0.001f && bright<0.001f) && (contrast>0.999f && contrast<1.001f) && (sat>0.999f && sat<1.001f);
  if(neutral) return; // identity per-clip grade: leave INB untouched.
  clSetKernelArg(kBright,0,sizeof(cl_mem),&g_buf[INB]); clSetKernelArg(kBright,1,sizeof(cl_mem),&g_buf[MID]); clSetKernelArg(kBright,2,sizeof(float),&bright); launch(kBright);
  clSetKernelArg(kContrast,0,sizeof(cl_mem),&g_buf[MID]); clSetKernelArg(kContrast,1,sizeof(cl_mem),&g_buf[INB]); clSetKernelArg(kContrast,2,sizeof(float),&contrast); launch(kContrast);
  if(sat<0.999f || sat>1.001f){ clSetKernelArg(kSat,0,sizeof(cl_mem),&g_buf[INB]); clSetKernelArg(kSat,1,sizeof(float),&sat); launch(kSat); }
}
// P2 LGG (3-way color wheels): per-channel out=clamp01(pow(clamp01(in*gain+lift),1/gamma)) IN PLACE
// on the grade-result buffer (OUTB), BEFORE look. Identity (lift 0 / gamma 1 / gain 1 on all three
// channels) is skipped (zero cost). White balance is already folded into the gains by the UI.
void fpx_gpu_lgg(float lr,float lg,float lb,float gar,float gag,float gab,float gnr,float gng,float gnb){
  if(!g_ready) return;
  int identity =
    (lr>-0.001f&&lr<0.001f)&&(lg>-0.001f&&lg<0.001f)&&(lb>-0.001f&&lb<0.001f)&&
    (gar>0.999f&&gar<1.001f)&&(gag>0.999f&&gag<1.001f)&&(gab>0.999f&&gab<1.001f)&&
    (gnr>0.999f&&gnr<1.001f)&&(gng>0.999f&&gng<1.001f)&&(gnb>0.999f&&gnb<1.001f);
  if(identity) return;
  clSetKernelArg(kLgg,0,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kLgg,1,sizeof(float),&lr); clSetKernelArg(kLgg,2,sizeof(float),&lg); clSetKernelArg(kLgg,3,sizeof(float),&lb);
  clSetKernelArg(kLgg,4,sizeof(float),&gar); clSetKernelArg(kLgg,5,sizeof(float),&gag); clSetKernelArg(kLgg,6,sizeof(float),&gab);
  clSetKernelArg(kLgg,7,sizeof(float),&gnr); clSetKernelArg(kLgg,8,sizeof(float),&gng); clSetKernelArg(kLgg,9,sizeof(float),&gnb);
  launch(kLgg);
}
// P2 BLUR: separable gaussian, 2 passes, IN PLACE on OUTB via g_tmp. sigma<=0 => no-op. Runs AFTER
// lgg, BEFORE look. Horizontal pass OUTB->g_tmp, vertical pass g_tmp->OUTB.
void fpx_gpu_blur(float sigma){
  if(!g_ready || sigma<=0.0f) return; // identity / no-op blur: leave OUTB untouched.
  clSetKernelArg(kBlurH,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kBlurH,1,sizeof(cl_mem),&g_tmp); clSetKernelArg(kBlurH,2,sizeof(float),&sigma); launch(kBlurH);
  clSetKernelArg(kBlurV,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kBlurV,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kBlurV,2,sizeof(float),&sigma); launch(kBlurV);
}
// P5 CURVE: a 5-point master tone curve, IN PLACE on OUTB. The identity (0,.25,.5,.75,1) is skipped
// so an un-curved clip is a true no-op. Runs AFTER blur, BEFORE look.
void fpx_gpu_curve(float y0,float y1,float y2,float y3,float y4){
  if(!g_ready) return;
  if(y0==0.0f && y1==0.25f && y2==0.5f && y3==0.75f && y4==1.0f) return; // identity
  clSetKernelArg(kCurve,0,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kCurve,1,sizeof(float),&y0); clSetKernelArg(kCurve,2,sizeof(float),&y1);
  clSetKernelArg(kCurve,3,sizeof(float),&y2); clSetKernelArg(kCurve,4,sizeof(float),&y3);
  clSetKernelArg(kCurve,5,sizeof(float),&y4); launch(kCurve);
}
// P6 SIMPLE-FX: in place on OUTB. kind 0 = skip (no-op default); 1 invert, 2 sepia, 3 grayscale,
// 4 posterize. Runs AFTER curve, BEFORE the look (first of the four P6 filters, per pinned order).
void fpx_gpu_simplefx(int kind){
  if(!g_ready || kind==0) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kSimplefx,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kSimplefx,1,sizeof(int),&kind);
  launch(kSimplefx);
}
// P6 VIGNETTE: in place on OUTB, radial edge darken by `amt`. amt<=0 = skip (no-op default). Runs
// after simple-fx, before sharpen (pinned P6 order: simplefx -> vignette -> sharpen -> flip).
void fpx_gpu_vignette(float amt){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kVignette,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kVignette,1,sizeof(float),&amt);
  launch(kVignette);
}
// P6 SHARPEN (unsharp): amt<=0 = skip. The kernel cannot read+write OUTB in place, so copy
// OUTB->g_tmp first (via k_copy, same convention as transform/blur), then sharpen g_tmp->OUTB.
void fpx_gpu_sharpen(float amt){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kSharpen,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kSharpen,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kSharpen,2,sizeof(float),&amt);
  launch(kSharpen);
}
// P6 FLIP (mirror): mode 0 = skip; 1 H (x->W-1-x), 2 V (y->H-1-y), 3 both. Copy OUTB->g_tmp first,
// then sample g_tmp at the flipped coord into OUTB. Last of the four P6 filters (before the look).
void fpx_gpu_flip(int mode){
  if(!g_ready || mode==0) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kFlip,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kFlip,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kFlip,2,sizeof(int),&mode);
  launch(kFlip);
}
// P7 HSL ADJUST: in place on OUTB. RGB->HSL, hue += hue_deg (wrap 360), saturation *= sat, lightness
// += light, HSL->RGB, clamp01. Identity hue_deg=0,sat=1,light=0 = skip (no-op default). Runs AFTER
// the P6 flip, BEFORE the look (first of the two P7 color filters). Per-pixel in place — no scratch.
void fpx_gpu_hsl(float hue_deg, float sat, float light){
  if(!g_ready) return;
  int identity = (hue_deg>-0.001f && hue_deg<0.001f) && (sat>0.999f && sat<1.001f) && (light>-0.001f && light<0.001f);
  if(identity) return; // identity HSL: leave OUTB untouched.
  clSetKernelArg(kHsl,0,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kHsl,1,sizeof(float),&hue_deg); clSetKernelArg(kHsl,2,sizeof(float),&sat); clSetKernelArg(kHsl,3,sizeof(float),&light);
  launch(kHsl);
}
// P7 LEVELS: in place on OUTB, per channel out=clamp01(pow(clamp01((c-in_black)/max(in_white-in_black,
// 1e-3)),1/max(gamma,1e-3))). Identity in_black=0,in_white=1,gamma=1 = skip (no-op default). Runs
// AFTER hsl, BEFORE the look (last of the two P7 color filters). Per-pixel in place — no scratch.
void fpx_gpu_levels(float in_black, float in_white, float gamma){
  if(!g_ready) return;
  int identity = (in_black>-0.001f && in_black<0.001f) && (in_white>0.999f && in_white<1.001f) && (gamma>0.999f && gamma<1.001f);
  if(identity) return; // identity levels: leave OUTB untouched.
  clSetKernelArg(kLevels,0,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kLevels,1,sizeof(float),&in_black); clSetKernelArg(kLevels,2,sizeof(float),&in_white); clSetKernelArg(kLevels,3,sizeof(float),&gamma);
  launch(kLevels);
}
// P8 MOSAIC (pixelate): block-average via top-left sampling. block<=1 = skip (no-op default, and the
// div-by-zero guard: block 0 and 1 both skip). The kernel cannot read+write OUTB in place (one source
// pixel per block read by many threads = read/write race), so copy OUTB->g_tmp first (via k_copy, same
// convention as transform/blur/sharpen/flip), then sample g_tmp's block top-left into OUTB. Runs AFTER
// the P7 levels, BEFORE the look (first of the two P8 filters, per pinned order).
void fpx_gpu_mosaic(int block){
  if(!g_ready || block<=1) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kMosaic,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kMosaic,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kMosaic,2,sizeof(int),&block);
  launch(kMosaic);
}
// P8 GRADIENT MAP: luma -> colour ramp, IN PLACE on OUTB (per-own-pixel, no scratch — like hsl/levels).
// luma=dot(rgb,[.299,.587,.114]); mapped=mix(lo,hi,luma); rgb=mix(rgb,mapped,amt). amt<=0 = skip
// (no-op default). Runs AFTER mosaic, BEFORE the look (last of the two P8 filters).
void fpx_gpu_gmap(float amt, float lo_r, float lo_g, float lo_b, float hi_r, float hi_g, float hi_b){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kGmap,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kGmap,1,sizeof(float),&amt);
  clSetKernelArg(kGmap,2,sizeof(float),&lo_r); clSetKernelArg(kGmap,3,sizeof(float),&lo_g); clSetKernelArg(kGmap,4,sizeof(float),&lo_b);
  clSetKernelArg(kGmap,5,sizeof(float),&hi_r); clSetKernelArg(kGmap,6,sizeof(float),&hi_g); clSetKernelArg(kGmap,7,sizeof(float),&hi_b);
  launch(kGmap);
}
// P9 DENOISE: edge-preserving 5x5 bilateral smooth blended by `strength`. strength<=0 = skip
// (no-op default). The kernel cannot read+write OUTB in place (a 5x5 neighbourhood read by many
// threads vs an in-place write = read/write race), so copy OUTB->g_tmp first (via k_copy, same
// convention as transform/blur/sharpen/flip/mosaic), then read g_tmp neighbours into OUTB. Runs
// AFTER the P8 gradient-map, BEFORE the look (first of the three P9 filters, per pinned order).
void fpx_gpu_denoise(float strength){
  if(!g_ready || strength<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kDenoise,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kDenoise,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kDenoise,2,sizeof(float),&strength);
  launch(kDenoise);
}
// P9 GLOW (bloom): bright-pass extract -> blur -> add back. amt<=0 = skip (no-op default). This runs
// as its OWN self-contained sequence using BOTH scratch buffers (g_tmp2 + g_tmp): extract the bright
// pass OUTB->g_tmp2, blur it at a FIXED sigma by REUSING the separable gaussian (kBlurH g_tmp2->g_tmp,
// kBlurV g_tmp->g_tmp2 — same arg order as fpx_gpu_blur), then combine OUTB = clamp01(OUTB +
// amt*g_tmp2). The blur ping-pongs g_tmp2->g_tmp->g_tmp2, so the final blurred bright pass lands back
// in g_tmp2 (what k_glow_combine reads). g_tmp is only borrowed mid-sequence and is not relied on
// after; OUTB is untouched until the combine. Runs after denoise, before rgb-shift.
void fpx_gpu_glow(float amt, float thr){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  float sigma=8.0f; // fixed bloom blur radius (~ceil(2*8)=16px each side, capped at 32 by the kernel)
  // 1) bright pass: OUTB (luma>thr ? rgb : 0) -> g_tmp2.
  clSetKernelArg(kGlowExtract,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kGlowExtract,1,sizeof(cl_mem),&g_tmp2); clSetKernelArg(kGlowExtract,2,sizeof(float),&thr);
  launch(kGlowExtract);
  // 2) blur the bright pass: horiz g_tmp2->g_tmp, vert g_tmp->g_tmp2 (final blurred pass in g_tmp2).
  clSetKernelArg(kBlurH,0,sizeof(cl_mem),&g_tmp2); clSetKernelArg(kBlurH,1,sizeof(cl_mem),&g_tmp); clSetKernelArg(kBlurH,2,sizeof(float),&sigma); launch(kBlurH);
  clSetKernelArg(kBlurV,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kBlurV,1,sizeof(cl_mem),&g_tmp2); clSetKernelArg(kBlurV,2,sizeof(float),&sigma); launch(kBlurV);
  // 3) combine: OUTB = clamp01(OUTB + amt*g_tmp2).
  clSetKernelArg(kGlowCombine,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kGlowCombine,1,sizeof(cl_mem),&g_tmp2); clSetKernelArg(kGlowCombine,2,sizeof(float),&amt);
  launch(kGlowCombine);
}
// P9 RGB-SHIFT (chromatic aberration): channel-offset sample by `px` pixels. px<=0 = skip (no-op
// default). The kernel cannot read+write OUTB in place (it samples neighbouring x columns), so copy
// OUTB->g_tmp first, then sample g_tmp's r at (x+off,y), b at (x-off,y), g/a at (x,y) into OUTB. The
// offset is rounded to the nearest integer pixel. Runs after glow, before the look (last P9 filter).
void fpx_gpu_rgbshift(float px){
  if(!g_ready || px<=0.0f) return; // no-op default: leave OUTB untouched.
  int off=(int)(px+0.5f); if(off<1) off=1; // rounded, at least 1px (px>0 here)
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kRgbshift,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kRgbshift,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kRgbshift,2,sizeof(int),&off);
  launch(kRgbshift);
}
// look: 0=none (final=out), 1=vhs, 2=lut3d -> final=look ; returns 1 if final is LOOK else 0
int fpx_gpu_look(int kind, float amt, int lut_n){
  if(!g_ready) return 0;
  if(kind==1){ clSetKernelArg(kVhs,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kVhs,1,sizeof(cl_mem),&g_buf[LOOKB]); clSetKernelArg(kVhs,2,sizeof(float),&amt); launch(kVhs); return 1; }
  if(kind==2){ clSetKernelArg(kLut,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kLut,1,sizeof(cl_mem),&g_buf[LOOKB]); clSetKernelArg(kLut,2,sizeof(cl_mem),&g_lut);
    clSetKernelArg(kLut,3,sizeof(int),&lut_n); clSetKernelArg(kLut,4,sizeof(float),&amt); launch(kLut); return 1; }
  return 0;
}
// download final buffer (0=out,1=look) -> host f32 (blocking)
void fpx_gpu_download_f32(int final_is_look, float* out){
  if(!g_ready||!out) return; int b = final_is_look?LOOKB:OUTB;
  clEnqueueReadBuffer(g_q,g_buf[b],CL_TRUE,0,GN*sizeof(float),out,0,NULL,NULL);
}
// download final -> host u8 (pack on GPU, then read)
void fpx_gpu_download_u8(int final_is_look, unsigned char* out){
  if(!g_ready||!out) return; int b = final_is_look?LOOKB:OUTB;
  clSetKernelArg(kPack,0,sizeof(cl_mem),&g_buf[b]); clSetKernelArg(kPack,1,sizeof(cl_mem),&g_stage); launch(kPack);
  clEnqueueReadBuffer(g_q,g_stage,CL_TRUE,0,GN*sizeof(unsigned char),out,0,NULL,NULL);
}
// scope: RGB histogram of the final buffer (0=out,1=look) -> out_hist[768] (R 0..255, G 256..511, B 512..767)
void fpx_gpu_histogram(int final_is_look, int* out_hist){
  if(!g_ready||!out_hist) return; int b = final_is_look?LOOKB:OUTB;
  clSetKernelArg(kHistClear,0,sizeof(cl_mem),&g_hist);
  size_t hg=768, hl=64; clEnqueueNDRangeKernel(g_q,kHistClear,1,NULL,&hg,&hl,0,NULL,NULL);
  clSetKernelArg(kHist,0,sizeof(cl_mem),&g_buf[b]); clSetKernelArg(kHist,1,sizeof(cl_mem),&g_hist); launch(kHist);
  clEnqueueReadBuffer(g_q,g_hist,CL_TRUE,0,768*sizeof(int),out_hist,0,NULL,NULL);
}
// scope helper: clear the 256x256 grid, accumulate (acc kernel over the frame), render to g_scope, read 256x256x4 u8
static void scope_render(int final_is_look, cl_kernel acc, cl_kernel img, float gain, unsigned char* out){
  int b = final_is_look?LOOKB:OUTB;
  clSetKernelArg(kGridClear,0,sizeof(cl_mem),&g_grid);
  size_t gg=65536, gl=64; clEnqueueNDRangeKernel(g_q,kGridClear,1,NULL,&gg,&gl,0,NULL,NULL);
  clSetKernelArg(acc,0,sizeof(cl_mem),&g_buf[b]); clSetKernelArg(acc,1,sizeof(cl_mem),&g_grid); launch(acc);
  clSetKernelArg(img,0,sizeof(cl_mem),&g_grid); clSetKernelArg(img,1,sizeof(cl_mem),&g_scope); clSetKernelArg(img,2,sizeof(float),&gain);
  size_t sg[2]={256,256}, sl[2]={8,8}; clEnqueueNDRangeKernel(g_q,img,2,NULL,sg,sl,0,NULL,NULL);
  clEnqueueReadBuffer(g_q,g_scope,CL_TRUE,0,256*256*4,out,0,NULL,NULL);
}
// luma waveform -> RGBA8 256x256 (out)
void fpx_gpu_waveform(int final_is_look, unsigned char* out){ if(g_ready&&out) scope_render(final_is_look,kWaveAcc,kWaveImg,0.06f,out); }
// vectorscope (U/V) -> RGBA8 256x256 (out)
void fpx_gpu_vectorscope(int final_is_look, unsigned char* out){ if(g_ready&&out) scope_render(final_is_look,kVecAcc,kVecImg,0.04f,out); }
// RGB parade (Triad-B P1): per-channel column waveform, 3 side-by-side panels (R|G|B) -> RGBA8
// 256x256 (out). Uses the dedicated 3*256*256 int parade grid: clear, accumulate all 3 channels over
// the frame, render the three panels into g_scope, read back. Same final-buffer selection as the
// other scopes (OUTB / LOOKB). Gain matches the waveform's so a busy frame fills similarly.
void fpx_gpu_parade(int final_is_look, unsigned char* out){
  if(!g_ready||!out) return;
  int b = final_is_look?LOOKB:OUTB;
  clSetKernelArg(kParadeClear,0,sizeof(cl_mem),&g_parade);
  size_t pg=3*65536, pl=64; clEnqueueNDRangeKernel(g_q,kParadeClear,1,NULL,&pg,&pl,0,NULL,NULL);
  clSetKernelArg(kParadeAcc,0,sizeof(cl_mem),&g_buf[b]); clSetKernelArg(kParadeAcc,1,sizeof(cl_mem),&g_parade); launch(kParadeAcc);
  float gain=0.06f;
  clSetKernelArg(kParadeImg,0,sizeof(cl_mem),&g_parade); clSetKernelArg(kParadeImg,1,sizeof(cl_mem),&g_scope); clSetKernelArg(kParadeImg,2,sizeof(float),&gain);
  size_t sg[2]={256,256}, sl[2]={8,8}; clEnqueueNDRangeKernel(g_q,kParadeImg,2,NULL,sg,sl,0,NULL,NULL);
  clEnqueueReadBuffer(g_q,g_scope,CL_TRUE,0,256*256*4,out,0,NULL,NULL);
}
void fpx_gpu_finish(void){ if(g_ready) clFinish(g_q); }
