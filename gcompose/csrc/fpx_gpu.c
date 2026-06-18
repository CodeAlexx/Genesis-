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
static cl_mem g_scope = NULL;     // 256*256*4 byte rendered scope image
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
// composite alpha-over: dst = over*aeff + base*(1-aeff), aeff=over.a*op ; alpha from base
"__kernel void k_composite(__global const float* base,__global const float* over,__global float* dst,float op){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float a=over[i+3]*op, inv=1.0f-a;\n"
"  dst[i+0]=over[i+0]*a+base[i+0]*inv; dst[i+1]=over[i+1]*a+base[i+1]*inv; dst[i+2]=over[i+2]*a+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
// picture-in-picture: shrink whole over into normalized rect [px,py,pw,ph]
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
"}\n";

// cached kernel objects (clCreateKernel once)
static cl_kernel K(const char* name){ cl_int e; cl_kernel k = clCreateKernel(g_prog, name, &e); return (e==CL_SUCCESS)?k:NULL; }
static cl_kernel kUnpack,kPack,kCopy,kComposite,kPip,kBright,kContrast,kSat,kLut,kVhs;
static cl_kernel kHistClear,kHist;
static cl_kernel kGridClear,kWaveAcc,kWaveImg,kVecAcc,kVecImg;
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
  g_stage = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY, GN*sizeof(unsigned char), NULL, &e); if(e!=CL_SUCCESS) return -9;
  g_hist  = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, 768*sizeof(int), NULL, &e); if(e!=CL_SUCCESS) return -11;
  g_grid  = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, 65536*sizeof(int), NULL, &e); if(e!=CL_SUCCESS) return -12;
  g_scope = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, 256*256*4, NULL, &e); if(e!=CL_SUCCESS) return -13;
  kUnpack=K("k_unpack"); kPack=K("k_pack"); kCopy=K("k_copy"); kComposite=K("k_composite"); kPip=K("k_pip");
  kBright=K("k_brightness"); kContrast=K("k_contrast"); kSat=K("k_saturation"); kLut=K("k_lut3d"); kVhs=K("k_vhs");
  kHistClear=K("k_hist_clear"); kHist=K("k_hist");
  kGridClear=K("k_grid_clear"); kWaveAcc=K("k_wave_acc"); kWaveImg=K("k_wave_img"); kVecAcc=K("k_vec_acc"); kVecImg=K("k_vec_img");
  kTrans[0]=K("k_crossfade"); kTrans[1]=K("k_wipe_lr"); kTrans[2]=K("k_wipe_rl"); kTrans[3]=K("k_wipe_up");
  kTrans[4]=K("k_wipe_down"); kTrans[5]=K("k_slide_lr"); kTrans[6]=K("k_zoom"); kTrans[7]=K("k_dissolve");
  if(!kUnpack||!kPack||!kComposite||!kPip||!kBright||!kContrast||!kLut||!kVhs||!kTrans[7]) return -10;
  // scope kernels must build too — else a SCOPE command later launches a NULL kernel (segfault).
  if(!kHistClear||!kHist||!kGridClear||!kWaveAcc||!kWaveImg||!kVecAcc||!kVecImg) return -14;
  g_ready=1; return 0;
}

// upload an RGBA8 frame to slot (0=base,1=over,2=trans): staging u8 -> unpack -> float buf
int fpx_gpu_upload_u8(int slot, const unsigned char* rgba8){
  if(!g_ready||slot<0||slot>2||!rgba8) return -1;
  if(clEnqueueWriteBuffer(g_q,g_stage,CL_FALSE,0,GN*sizeof(unsigned char),rgba8,0,NULL,NULL)!=CL_SUCCESS) return -2;
  clSetKernelArg(kUnpack,0,sizeof(cl_mem),&g_stage); clSetKernelArg(kUnpack,1,sizeof(cl_mem),&g_buf[slot]);
  return launch(kUnpack);
}
int fpx_gpu_upload_f32(int slot, const float* f32){
  if(!g_ready||slot<0||slot>2||!f32) return -1;
  return clEnqueueWriteBuffer(g_q,g_buf[slot],CL_FALSE,0,GN*sizeof(float),f32,0,NULL,NULL)==CL_SUCCESS?0:-2;
}
int fpx_gpu_upload_lut(const float* lut, int nfloats){
  if(!g_ready||!lut||nfloats<=0||nfloats>MAXLUTF) return -1;
  return clEnqueueWriteBuffer(g_q,g_lut,CL_FALSE,0,nfloats*sizeof(float),lut,0,NULL,NULL)==CL_SUCCESS?0:-2;
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
void fpx_gpu_finish(void){ if(g_ready) clFinish(g_q); }
