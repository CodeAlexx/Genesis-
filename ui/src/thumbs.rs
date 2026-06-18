//! Lazy, process-global cache for timeline visuals: per-clip thumbnails + audio envelopes.
//!
//! SLICE C (timeline visuals). The timeline widget wants two cheap-to-draw things on each
//! clip: small in/out-point thumbnails for video, and a peak envelope for audio. Both come
//! from the `gcompose` engine worker via the PINNED `worker` APIs — egui never touches
//! FFmpeg/OpenCL directly. Decoding/scanning media is expensive, so everything here is
//! memoised:
//!   - thumbnails keyed by (media index, source frame) -> uploaded `egui::TextureHandle`,
//!   - envelopes keyed by media index -> `Vec<f32>` of per-bucket peaks (0..1).
//!
//! A process-global `OnceLock<Mutex<Cache>>` (mirroring `worker::WORKER`) lets the timeline
//! widget reach the cache without threading state through the PINNED `timeline_ui` signature.
//! Access it from inside the widget via `ui.ctx()` (a texture upload needs a live context).
//!
//! Fetch discipline (so the single serial worker is not hammered): the timeline only ever
//! asks for the in-point + out-point thumbnail per clip and ONE envelope per media. On a
//! decode/scan failure we still insert a sentinel (a `None` for thumbs, an empty `Vec` for
//! envelopes) so a permanently-undecodable media is not retried every frame.

use crate::worker;
use eframe::egui;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Thumbnail dimensions (pixels). Small — these are blitted into ~40px-tall timeline clips.
pub const TW: usize = 80;
pub const TH: usize = 45;

/// Per-media envelope resolution (peak buckets). Matches MojoMedia's WAVEK ballpark; plenty
/// of detail for a clip that is at most a few hundred px wide on the timeline.
pub const ENV_BUCKETS: usize = 480;

/// Memoised timeline visuals. Keys reference media by its index in `Project.media` so a clip
/// move/trim never invalidates an entry (only the in/out *source frame* selects the thumb).
pub struct Cache {
    /// (media index, source frame) -> uploaded texture. `None` = decode failed (cached so we
    /// do not refetch a bad frame every repaint). `Some` holds the live texture handle.
    thumbs: HashMap<(usize, i64), Option<egui::TextureHandle>>,
    /// media index -> per-bucket peak envelope (0..1). An empty `Vec` is the failure sentinel
    /// (no audio stream / decode error) so we likewise do not refetch.
    envs: HashMap<usize, Vec<f32>>,
}

impl Cache {
    fn new() -> Cache {
        Cache { thumbs: HashMap::new(), envs: HashMap::new() }
    }

    /// Fetch (or recall) the thumbnail for `media_path` at source `frame`, returning a texture
    /// id ready for `painter.image`. On a cache miss we ask the worker to decode one RGBA frame
    /// (`worker::thumbnail`), upload it (`worker::rgba_to_texture`), and store the handle. The
    /// handle is kept alive in the cache for the life of the process, so the returned `TextureId`
    /// stays valid across repaints. Returns `None` when the frame cannot be decoded (and caches
    /// that miss as a sentinel so it is not retried).
    pub fn thumb(
        &mut self,
        ctx: &egui::Context,
        media_idx: usize,
        media_path: &str,
        frame: i64,
    ) -> Option<egui::TextureId> {
        let key = (media_idx, frame.max(0));
        if let Some(slot) = self.thumbs.get(&key) {
            return slot.as_ref().map(|h| h.id());
        }
        // Miss: decode one frame off the worker and upload it.
        let handle = worker::thumbnail(media_path, key.1, TW, TH)
            .filter(|buf| buf.len() == TW * TH * 4)
            .map(|buf| rgba_to_thumb_texture(ctx, &buf));
        let id = handle.as_ref().map(|h| h.id());
        self.thumbs.insert(key, handle);
        id
    }

    /// Fetch (or recall) the per-bucket peak envelope for `media_path`. On a miss we ask the
    /// worker to scan the audio (`worker::audio_envelope`); on failure we cache an empty `Vec`
    /// so a media with no audio (or an undecodable one) is not rescanned every repaint. Always
    /// returns a borrowed slice (possibly empty) — the caller treats empty as "draw nothing".
    pub fn envelope(&mut self, media_idx: usize, media_path: &str, buckets: usize) -> &[f32] {
        // entry().or_insert_with avoids a double lookup; the closure only runs on a miss.
        self.envs
            .entry(media_idx)
            .or_insert_with(|| worker::audio_envelope(media_path, buckets).unwrap_or_default())
            .as_slice()
    }
}

/// Upload a TW×TH RGBA8 buffer as a timeline thumbnail texture (linear filtering — these are
/// scaled down into narrow clips). Mirrors `worker::rgba_to_texture` but at thumbnail size.
///
/// NOTE: the `"tl_thumb"` string is only a debug label — `Context::load_texture` passes it to
/// `tex_mngr.alloc()`, which mints a brand-new `TextureId` on EVERY call regardless of the name,
/// so names never alias and reusing one here does NOT deduplicate textures. What actually keeps
/// each thumbnail valid and distinct is that its `TextureHandle` is retained in `Cache.thumbs`
/// for the life of the process: `TextureHandle`'s Drop frees the GPU texture only when the last
/// handle drops, so a cached handle's `TextureId` stays live across repaints. Do NOT try to
/// "deduplicate by name" — the cache key (media index, source frame) is the real dedup.
fn rgba_to_thumb_texture(ctx: &egui::Context, buf: &[u8]) -> egui::TextureHandle {
    let img = egui::ColorImage::from_rgba_unmultiplied([TW, TH], buf);
    ctx.load_texture("tl_thumb", img, egui::TextureOptions::LINEAR)
}

/// Process-global cache, lazily initialised on first use (same pattern as `worker::WORKER`).
static THUMBS: OnceLock<Mutex<Cache>> = OnceLock::new();

fn cache_slot() -> &'static Mutex<Cache> {
    THUMBS.get_or_init(|| Mutex::new(Cache::new()))
}

/// Run `f` with exclusive access to the global cache. The closure does the fetch/draw-prep so
/// the lock is held only briefly. Returns `f`'s result; on a poisoned lock the closure is not
/// run and `None` is returned (the timeline simply skips visuals that frame — never panics).
pub fn with_cache<R>(f: impl FnOnce(&mut Cache) -> R) -> Option<R> {
    cache_slot().lock().ok().map(|mut c| f(&mut c))
}
