//! Client to the `gcompose` engine worker (separate process; owns OpenCL — the egui process
//! never links/calls OpenCL).
//!
//! SHARED CONTRACT (owned by the engine/worker team). Today: spawns `gcompose` per request
//! (with a retry to absorb its small OpenCL-init flake). P1 replaces this with a persistent
//! worker + a per-frame CompositeSpec over a pipe + shared-memory readback.

use crate::model::Project;
use eframe::egui;

pub const PVW: usize = 1280;
pub const PVH: usize = 856;
const OVER_PATH: &str = "/tmp/pip_v2.mp4"; // demo overlay until the model drives layers
const PREVIEW_RGBA: &str = "/tmp/genesis_preview.rgba";

fn worker_path() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join("gcompose"))
}

/// Compose the program at frame `t` -> RGBA8 PVW*PVH. Retries the worker spawn a few times.
/// NOTE (P1): currently composes base = project.media[0] + the demo overlay, ignoring `t` and
/// the real timeline. The persistent-worker + frame-spec protocol replaces this.
pub fn request_frame(project: &Project, _t: i64) -> Option<Vec<u8>> {
    let base = project.media.first()?;
    let w = worker_path()?;
    for attempt in 0..4 {
        let ok = std::process::Command::new(&w)
            .arg(base)
            .arg(OVER_PATH)
            .arg(PREVIEW_RGBA)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            if let Ok(bytes) = std::fs::read(PREVIEW_RGBA) {
                if bytes.len() == PVW * PVH * 4 {
                    return Some(bytes);
                }
            }
        }
        eprintln!("gcompose attempt {} failed; retrying", attempt + 1);
    }
    None
}

/// Upload an RGBA8 PVW×PVH buffer as an egui texture (GL — needs a live context).
pub fn rgba_to_texture(ctx: &egui::Context, buf: &[u8]) -> egui::TextureHandle {
    let img = egui::ColorImage::from_rgba_unmultiplied([PVW, PVH], buf);
    ctx.load_texture("preview", img, egui::TextureOptions::LINEAR)
}
