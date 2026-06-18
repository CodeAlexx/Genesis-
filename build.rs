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

    // 2) Compile the vendored shim into a static lib. .compile() emits its link directive HERE.
    let mut build = cc::Build::new();
    build.file("csrc/fpx_decode.c").opt_level(2).warnings(false);
    for inc in &includes {
        build.include(inc);
    }
    build.compile("fpxengine");

    // 3) NOW emit the FFmpeg link directives — AFTER the static lib, so the linker resolves
    //    fpxengine's references to av_* against libraries that come later on the link line.
    for lib in libs {
        pkg_config::probe_library(lib)
            .unwrap_or_else(|e| panic!("pkg-config link probe failed for {lib}: {e}"));
    }

    println!("cargo:rerun-if-changed=csrc/fpx_decode.c");
    println!("cargo:rerun-if-changed=build.rs");
}
