// build.rs — compile the vendored C engine shims and link FFmpeg.
//
// Genesis keeps the verified C engine from MojoMedia (FFmpeg decode/encode/audio +
// the OpenCL compute shim) and only rewrites the app/UI layer in Rust. Phase 0 vendors
// just fpx_decode.c; later phases add fpx_gpu.c (OpenCL), fpx_encode.c, fpx_audio.c.

fn main() {
    let libs = ["libavformat", "libavcodec", "libswscale", "libavutil"];

    // 1) Probe for INCLUDE paths only — suppress cargo link metadata so we control link order.
    let mut includes = Vec::new();
    for lib in libs {
        let p = pkg_config::Config::new()
            .cargo_metadata(false)
            .probe(lib)
            .unwrap_or_else(|e| panic!("pkg-config failed for {lib}: {e}"));
        includes.extend(p.include_paths);
    }

    // OpenCL headers (CL/cl.h) ship with CUDA on this box; libOpenCL.so is in the CUDA lib dir.
    let cl_inc = "/usr/local/cuda/include";
    let cl_lib = "/usr/local/cuda/targets/x86_64-linux/lib";

    // 2) Compile the vendored shims into one static lib. .compile() emits its link directive HERE.
    let mut build = cc::Build::new();
    build
        .file("csrc/fpx_decode.c")
        .file("csrc/fpx_gpu.c") // OpenCL compute shim (composite/grade/pip/look/scopes)
        .opt_level(2)
        .warnings(false)
        .include(cl_inc);
    for inc in &includes {
        build.include(inc);
    }
    build.compile("fpxengine");

    // 3) NOW emit the link directives — AFTER the static lib, so fpxengine's references resolve
    //    against libraries that come later on the link line.
    for lib in libs {
        pkg_config::probe_library(lib)
            .unwrap_or_else(|e| panic!("pkg-config link probe failed for {lib}: {e}"));
    }
    println!("cargo:rustc-link-search=native={cl_lib}");
    println!("cargo:rustc-link-lib=dylib=OpenCL");
    println!("cargo:rustc-link-lib=dylib=m");

    println!("cargo:rerun-if-changed=csrc/fpx_decode.c");
    println!("cargo:rerun-if-changed=csrc/fpx_gpu.c");
    println!("cargo:rerun-if-changed=build.rs");
}
