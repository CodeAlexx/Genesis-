//! Real Shotcut-dark PNG icon system (Roadmap P10).
//!
//! SLICE B (icons). A single pre-baked blob — `ui/assets/icons_dark_32.rgba` — holds 39
//! contiguous 32×32 RGBA8 icons, icon `i` starting at byte `i * 32 * 32 * 4 = i * 4096`
//! (total 39 * 4096 = 159744 bytes). The toolbar (and any later widget) asks for an icon by
//! its lowercase name (`"play"`, `"open"`, …) and gets back an `egui::TextureId` ready for
//! `painter.image` / `egui::Image::new` — or `None` so the caller can fall back to text.
//!
//! Two process-global lazy caches, mirroring the `worker::WORKER` / `thumbs::THUMBS` pattern:
//!   - `BLOB`: the raw bytes of the icon file, read from disk exactly once (or `None` if the
//!     blob can't be located — every `icon()` then returns `None` and callers fall back).
//!   - `ICONS`: index -> uploaded `egui::TextureHandle`. The handle is RETAINED for the life of
//!     the process so its `TextureId` stays valid across repaints (a `TextureHandle`'s Drop is
//!     what frees the GPU texture — see the same discipline in `thumbs.rs`).
//!
//! Resolution: each 32×32 icon is 4096 bytes; `ColorImage::from_rgba_unmultiplied([32,32], &slice)`
//! → `ctx.load_texture` → cache. Unknown names and a missing blob both return `None`.

use eframe::egui;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Icon dimensions (pixels). Each icon in the blob is `DIM × DIM` RGBA8.
const DIM: usize = 32;

/// Bytes per icon: `DIM * DIM * 4`.
const ICON_BYTES: usize = DIM * DIM * 4; // 4096

/// Number of icons in the blob (kept for a defensive bounds check on the blob length).
const ICON_COUNT: usize = 39;

/// Total expected blob size, used to reject a truncated / wrong file.
const BLOB_BYTES: usize = ICON_COUNT * ICON_BYTES; // 159744

/// Blob filename, relative to whichever candidate directory we search.
const BLOB_NAME: &str = "icons_dark_32.rgba";

/// Map a lowercase icon name to its index in the blob. Mirrors `ui/assets/icons_index.txt`
/// (the GENERATED index table) and the wave contract's pinned name list exactly. Returns
/// `None` for an unknown name so `icon()` can fall back to text.
fn name_to_index(name: &str) -> Option<usize> {
    let i = match name {
        "play" => 0,
        "pause" => 1,
        "stop" => 2,
        "seek_back" => 3,
        "seek_fwd" => 4,
        "skip_back" => 5,
        "skip_fwd" => 6,
        "loop" => 7,
        "record" => 8,
        "cut" => 9,
        "copy" => 10,
        "paste" => 11,
        "delete" => 12,
        "undo" => 13,
        "redo" => 14,
        "split" => 15,
        "slice" => 16,
        "lift" => 17,
        "ripple" => 18,
        "overwrite" => 19,
        "add" => 20,
        "remove" => 21,
        "locked" => 22,
        "unlocked" => 23,
        "visible" => 24,
        "hidden" => 25,
        "muted" => 26,
        "volume" => 27,
        "zoom_in" => 28,
        "zoom_out" => 29,
        "zoom_fit" => 30,
        "snap" => 31,
        "marker" => 32,
        "new" => 33,
        "open" => 34,
        "save" => 35,
        "export" => 36,
        "menu" => 37,
        "color_pick" => 38,
        _ => return None,
    };
    Some(i)
}

/// Candidate directories that might contain the icon blob, in priority order:
///   1. the directory holding the running executable (deployed layout: blob beside the binary),
///   2. `<exe_dir>/assets`,
///   3. `<exe_dir>/../assets`, `<exe_dir>/../../assets` (target/debug or target/release → repo),
///   4. the source-tree `ui/assets` relative to the current working directory (dev `cargo run`).
/// We return full candidate *file* paths (dir + BLOB_NAME) and try each until one reads.
fn candidate_paths() -> Vec<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            dirs.push(exe_dir.to_path_buf());
            dirs.push(exe_dir.join("assets"));
            // Walk a couple of levels up (target/debug, target/release) into a sibling assets dir.
            let mut up = exe_dir.to_path_buf();
            for _ in 0..3 {
                if let Some(parent) = up.parent() {
                    up = parent.to_path_buf();
                    dirs.push(up.join("assets"));
                    dirs.push(up.join("ui").join("assets"));
                } else {
                    break;
                }
            }
        }
    }

    // Dev fallbacks relative to the working directory.
    dirs.push(std::path::PathBuf::from("ui/assets"));
    dirs.push(std::path::PathBuf::from("assets"));

    dirs.into_iter().map(|d| d.join(BLOB_NAME)).collect()
}

/// Process-global, read-once raw icon blob. `Some(bytes)` once located; `None` if no candidate
/// path yields a blob of the expected size. Lazily initialised on first `icon()` call.
static BLOB: OnceLock<Option<Vec<u8>>> = OnceLock::new();

/// Load (once) the raw icon blob from the first candidate path that reads as exactly
/// `BLOB_BYTES` bytes. A wrong-sized file is rejected (treated as "no blob") so a truncated /
/// stale asset can't slice garbage. Returns a borrow of the cached `Option<Vec<u8>>`.
fn blob() -> Option<&'static [u8]> {
    BLOB.get_or_init(|| {
        for path in candidate_paths() {
            if let Ok(bytes) = std::fs::read(&path) {
                if bytes.len() == BLOB_BYTES {
                    return Some(bytes);
                }
                // Wrong size: keep trying other candidates rather than slicing a bad file.
                eprintln!(
                    "icons: {} has {} bytes, expected {} — skipping",
                    path.display(),
                    bytes.len(),
                    BLOB_BYTES
                );
            }
        }
        eprintln!("icons: no usable {} found (toolbar will fall back to text)", BLOB_NAME);
        None
    })
    .as_deref()
}

/// Process-global cache of uploaded icon textures, keyed by blob index. Same lazy
/// `OnceLock<Mutex<HashMap>>` pattern as `thumbs::THUMBS`. The `TextureHandle` is retained for
/// the life of the process so its `TextureId` stays valid across repaints.
static ICONS: OnceLock<Mutex<HashMap<usize, egui::TextureHandle>>> = OnceLock::new();

fn icon_cache() -> &'static Mutex<HashMap<usize, egui::TextureHandle>> {
    ICONS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Resolve a lowercase icon `name` to an `egui::TextureId` ready for `painter.image` /
/// `egui::Image::new`, uploading + caching the icon on first use. Returns `None` for an unknown
/// name or when the icon blob can't be located — callers should fall back to a text label.
///
/// PINNED API (consumed by Slice C): `pub fn icon(ctx, name) -> Option<egui::TextureId>`.
pub fn icon(ctx: &egui::Context, name: &str) -> Option<egui::TextureId> {
    let index = name_to_index(name)?;

    let cache = icon_cache();
    let mut map = cache.lock().ok()?;

    // Cache hit: the retained handle keeps the TextureId valid.
    if let Some(handle) = map.get(&index) {
        return Some(handle.id());
    }

    // Miss: slice the icon's 4096-byte run out of the blob and upload it.
    let bytes = blob()?;
    let start = index * ICON_BYTES;
    let end = start + ICON_BYTES;
    let slice = bytes.get(start..end)?; // defensive: never panic on a short blob.

    let img = egui::ColorImage::from_rgba_unmultiplied([DIM, DIM], slice);
    let handle = ctx.load_texture(format!("icon_{name}"), img, egui::TextureOptions::LINEAR);
    let id = handle.id();
    map.insert(index, handle);
    Some(id)
}
