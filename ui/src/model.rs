//! Project data model — the Rust win over MojoMedia's flat parallel lists: real structs.
//!
//! SHARED CONTRACT. Owned by the timeline/model team; consumed by worker + app + pool.
//! Frame units are timeline frames (30 fps assumed for now).

#[derive(Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Clip {
    pub media: usize,  // index into Project.media
    pub src_in: i64,   // source in-point (frames)
    pub len: i64,      // length on the timeline (frames)
    pub t0: i64,       // timeline start (frames)
    pub track: u8,     // 0 = V1, 1 = V2, 2 = A1
    pub look: i32,     // per-clip LOOK index (0 = none)
    pub look_amt: f32, // look mix 0..1
    pub fade_in: i64,
    pub fade_out: i64,
    pub px: f32, // PiP rect (fractions of frame)
    pub py: f32,
    pub pw: f32,
    pub ph: f32,
    // Per-clip LUT path for look == 2 (LUT3D); empty = none. PINNED this wave: produced here
    // (Team B), consumed by Team A (engine look — loaded via fpx_load_cube + uploaded with
    // fpx_gpu_upload_lut, cached per path) and Team C (Look picker UI). `#[serde(default)]` so
    // pre-LUT .json projects still deserialize (the field defaults to "" = no LUT). Clip.look
    // semantics: 0 = None, 1 = VHS, 2 = LUT3D (uses this `lut`).
    #[serde(default)]
    pub lut: String,
}

impl Clip {
    pub fn video(media: usize, t0: i64, len: i64, track: u8, name_hint: &str) -> Clip {
        let _ = name_hint;
        Clip { media, src_in: 0, len, t0, track, look: 0, look_amt: 1.0, fade_in: 0, fade_out: 0, px: 0.0, py: 0.0, pw: 1.0, ph: 1.0, lut: String::new() }
    }
    pub fn end(&self) -> i64 {
        self.t0 + self.len
    }
}

/// One keyframe on a scalar track: `v` is the value at timeline (or clip-local) frame `t`.
/// Mirrors MojoMedia's parallel `KfTrack { frames, values }` but as a real struct (the Rust
/// win): a `Vec<Kf>` kept sorted ascending by `t` replaces the two parallel lists.
#[derive(Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Kf {
    pub t: i64,
    pub v: f32,
}

/// One per-clip PiP keyframe, stored flat (mirrors MojoMedia `PipKf`): which clip, which
/// param (0=px,1=py,2=pw,3=ph), the CLIP-LOCAL frame, and the value. Flat storage (one Vec
/// for the whole project) is chosen over a Vec-per-clip so the set survives split/delete
/// without re-indexing nested vectors; see `remap_clip_keys` for the index-stability policy.
#[derive(Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PipKey {
    pub clip: usize, // clip index these keys animate
    pub par: u8,     // 0 = px, 1 = py, 2 = pw, 3 = ph
    pub t_local: i64, // clip-local frame (t - clip.t0)
    pub v: f32,
}

/// Linear keyframe eval shared by grade + PiP: value of a sorted-ascending `Vec<Kf>` at `t`,
/// or `fallback` when the track is empty. Clamps to the first/last value outside the range.
/// Formula matches MojoMedia kf_eval / pip_eval: `blend*(vb-va)+va`.
fn eval_track(track: &[Kf], t: i64, fallback: f32) -> f32 {
    let n = track.len();
    if n == 0 {
        return fallback;
    }
    if t <= track[0].t {
        return track[0].v;
    }
    if t >= track[n - 1].t {
        return track[n - 1].v;
    }
    // find i such that track[i].t <= t < track[i+1].t
    let mut i = 0;
    while i < n - 1 && track[i + 1].t <= t {
        i += 1;
    }
    let fa = track[i].t;
    let fb = track[i + 1].t;
    let blend = (t - fa) as f64 / (fb - fa) as f64;
    (blend * (track[i + 1].v - track[i].v) as f64) as f32 + track[i].v
}

/// Sorted insert-or-replace into a `Vec<Kf>` keyed on `t` (mirrors MojoMedia kf_set): if a
/// key already exists at `t` its value is overwritten, otherwise the key is inserted so the
/// track stays ascending in `t`.
fn set_track(track: &mut Vec<Kf>, t: i64, v: f32) {
    match track.binary_search_by(|k| k.t.cmp(&t)) {
        Ok(idx) => track[idx].v = v,        // replace at existing frame
        Err(idx) => track.insert(idx, Kf { t, v }), // sorted insert
    }
}

#[derive(Clone, Default)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Project {
    pub media: Vec<String>, // media file paths; clips index into this
    pub names: Vec<String>, // display names per media
    pub clips: Vec<Clip>,
    #[serde(default)]
    pub trans: Vec<i32>, // transition id per boundary (-1 = none)
    #[serde(default)]
    pub bright: f32,
    #[serde(default)]
    pub contrast: f32,
    #[serde(default)]
    pub sat: f32,
    #[serde(default)]
    pub markers: Vec<i64>, // timeline markers (frames); the scrub/playhead can snap to them

    // ----- keyframe storage (Slice C; all #[serde(default)] so pre-keyframe .json loads) -----
    // Program-wide grade tracks, each a Vec<Kf> sorted ascending by t (timeline frames). An
    // EMPTY track means "use the static bright/contrast/sat field" — grade_at() falls back to
    // the static value so the existing non-animated grade keeps working unchanged. opacity_kf
    // is reserved for a future V2-opacity animation (worker does not yet read it; harmless to
    // store). Consumed by Team A worker::resolve_frame via grade_at(t).
    #[serde(default)]
    pub bright_kf: Vec<Kf>,
    #[serde(default)]
    pub contrast_kf: Vec<Kf>,
    #[serde(default)]
    pub sat_kf: Vec<Kf>,
    #[serde(default)]
    pub opacity_kf: Vec<Kf>,

    // Per-clip PiP keyframes, flat (mirrors MojoMedia PipKf). Each entry binds (clip, param,
    // clip-local frame) -> value. An (clip, param) with NO entries falls back to that clip's
    // static px/py/pw/ph in pip_at(). Consumed by Team A worker::resolve_frame via pip_at().
    #[serde(default)]
    pub pip_kf: Vec<PipKey>,

    // ----- per-track state (PINNED this wave; index 0 = V1, 1 = V2, 2 = A1) -----
    // serde(default) so older .json projects (without these keys) still deserialize to
    // [false; 3]. These fields are exposed for the worker via is_hidden()/is_muted()/
    // is_locked() below. INTEGRATOR WIRING REQUIRED (Team A): worker.rs does NOT yet
    // consult them — resolve_frame() composites by Clip.track alone and build_audio_lines()
    // gates on a static track_is_audible(track). Until Team A calls project.is_hidden(track)
    // in resolve_frame (skip a hidden VIDEO track) and project.is_muted(track) in
    // build_audio_lines (drop a muted track's audio), these toggles change neither the
    // preview nor the export. track_lock is advisory this wave (edits to a locked track
    // should be blocked; timeline.rs already calls is_locked()).
    #[serde(default)]
    pub track_hide: [bool; 3], // true => that VIDEO track is not shown/composited
    #[serde(default)]
    pub track_mute: [bool; 3], // true => that track contributes NO audio to the render
    #[serde(default)]
    pub track_lock: [bool; 3], // true => edits to that track are blocked (advisory)
}

impl Project {
    /// A demo project (3 clips) used until the media pool + import land.
    pub fn demo(media: String) -> Project {
        Project {
            media: vec![media],
            names: vec!["clip".into()],
            clips: vec![
                Clip::video(0, 0, 120, 0, "intro"),
                Clip::video(0, 70, 90, 1, "overlay"),
                Clip::video(0, 0, 160, 2, "audio"),
            ],
            trans: vec![],
            bright: 0.0,
            contrast: 1.0,
            sat: 1.0,
            markers: vec![],
            track_hide: [false; 3],
            track_mute: [false; 3],
            track_lock: [false; 3],
            bright_kf: vec![],
            contrast_kf: vec![],
            sat_kf: vec![],
            opacity_kf: vec![],
            pip_kf: vec![],
        }
    }

    pub fn total_frames(&self) -> i64 {
        self.clips.iter().map(|c| c.end()).max().unwrap_or(1).max(1)
    }

    // ----- keyframe eval (PINNED; consumed by Team A worker::resolve_frame) -------------
    // Both methods return the STATIC field when the relevant track has no keys, so a project
    // with no keyframes behaves exactly as before (worker can call these unconditionally).

    /// (bright, contrast, sat) at timeline frame `t`. Each component linearly interpolates its
    /// keyframe track; an empty track falls back to the static `bright`/`contrast`/`sat` field.
    pub fn grade_at(&self, t: i64) -> (f32, f32, f32) {
        (
            eval_track(&self.bright_kf, t, self.bright),
            eval_track(&self.contrast_kf, t, self.contrast),
            eval_track(&self.sat_kf, t, self.sat),
        )
    }

    /// (px, py, pw, ph) for clip `clip_idx` at CLIP-LOCAL frame `t_local`. Each param linearly
    /// interpolates its per-clip PiP keyframes; a param with no keys for this clip falls back to
    /// the clip's static `px`/`py`/`pw`/`ph`. An out-of-range `clip_idx` returns the full-frame
    /// default (0,0,1,1) so the worker never panics on a stale index.
    pub fn pip_at(&self, clip_idx: usize, t_local: i64) -> (f32, f32, f32, f32) {
        let (sx, sy, sw, sh) = match self.clips.get(clip_idx) {
            Some(c) => (c.px, c.py, c.pw, c.ph),
            None => (0.0, 0.0, 1.0, 1.0),
        };
        (
            self.eval_pip(clip_idx, 0, t_local, sx),
            self.eval_pip(clip_idx, 1, t_local, sy),
            self.eval_pip(clip_idx, 2, t_local, sw),
            self.eval_pip(clip_idx, 3, t_local, sh),
        )
    }

    /// Interpolate one (clip, param) PiP track from the flat `pip_kf` store at clip-local frame
    /// `t`. Mirrors MojoMedia pip_eval: scan the flat list for matching (clip,par) entries,
    /// track the nearest key at/below `t` (lo) and above `t` (hi), then linearly blend; empty
    /// -> `fallback`, clamp to lo/hi at the ends. The flat list is unsorted, so this is an O(n)
    /// scan (n = total PiP keys, small) rather than a binary search.
    fn eval_pip(&self, clip: usize, par: u8, t: i64, fallback: f32) -> f32 {
        let mut lo: Option<(i64, f32)> = None;
        let mut hi: Option<(i64, f32)> = None;
        for k in self.pip_kf.iter().filter(|k| k.clip == clip && k.par == par) {
            if k.t_local <= t {
                if lo.is_none_or(|(lf, _)| k.t_local > lf) {
                    lo = Some((k.t_local, k.v));
                }
            } else if hi.is_none_or(|(hf, _)| k.t_local < hf) {
                hi = Some((k.t_local, k.v));
            }
        }
        match (lo, hi) {
            (None, None) => fallback,
            (Some((_, lv)), None) => lv,          // clamp after the last key
            (None, Some((_, hv))) => hv,          // clamp before the first key
            (Some((lf, lv)), Some((hf, hv))) => {
                let blend = (t - lf) as f64 / (hf - lf) as f64;
                (blend * (hv - lv) as f64) as f32 + lv
            }
        }
    }

    // ----- keyframe edit ops (Slice C; called by panels::properties_ui Key buttons) -----

    /// Snapshot the CURRENT static grade (bright/contrast/sat) into a keyframe at timeline
    /// frame `t` on all three grade tracks (sorted insert-or-replace). Mirrors MojoMedia's
    /// "K" buttons keying brightness/contrast/saturation at the playhead with the live values.
    pub fn add_grade_key(&mut self, t: i64) {
        let (b, c, s) = (self.bright, self.contrast, self.sat);
        set_track(&mut self.bright_kf, t, b);
        set_track(&mut self.contrast_kf, t, c);
        set_track(&mut self.sat_kf, t, s);
    }

    /// Snapshot clip `clip_idx`'s CURRENT static PiP rect (px/py/pw/ph) into PiP keyframes at
    /// CLIP-LOCAL frame `t_local` (one per param 0..3, insert-or-replace). Mirrors MojoMedia's
    /// "Key PiP" button keying all four params at the clip-local frame. No-op for a bad index.
    pub fn add_pip_key(&mut self, clip_idx: usize, t_local: i64) {
        let (px, py, pw, ph) = match self.clips.get(clip_idx) {
            Some(c) => (c.px, c.py, c.pw, c.ph),
            None => return,
        };
        self.set_pip(clip_idx, 0, t_local, px);
        self.set_pip(clip_idx, 1, t_local, py);
        self.set_pip(clip_idx, 2, t_local, pw);
        self.set_pip(clip_idx, 3, t_local, ph);
    }

    /// Insert-or-replace a single PiP keyframe in the flat store (mirrors MojoMedia pip_set):
    /// overwrite the value if an entry already exists for (clip, par, t_local), else append.
    fn set_pip(&mut self, clip: usize, par: u8, t_local: i64, v: f32) {
        if let Some(k) = self
            .pip_kf
            .iter_mut()
            .find(|k| k.clip == clip && k.par == par && k.t_local == t_local)
        {
            k.v = v;
        } else {
            self.pip_kf.push(PipKey { clip, par, t_local, v });
        }
    }

    /// Count of PiP keyframes bound to clip `clip_idx` (any param) — for the panel's key-count
    /// readout. O(n) over the small flat store.
    pub fn pip_key_count(&self, clip_idx: usize) -> usize {
        self.pip_kf.iter().filter(|k| k.clip == clip_idx).count()
    }

    // ----- keyframe DRAG/DELETE edit ops (Slice B; called by timeline diamond/tick drags) -----
    // `track` selects the grade keyframe track: 0 = bright_kf, 1 = contrast_kf, 2 = sat_kf,
    // 3 = opacity_kf. All ops bounds-check (bad track / idx -> no-op) so a stale index from a
    // mid-drag mutation can never panic. PiP ops address the flat `pip_kf` store by index.

    /// Mutable borrow of one grade track by its PINNED index, or `None` for an out-of-range
    /// track. Internal helper for the move/delete ops below.
    fn grade_track_mut(&mut self, track: u8) -> Option<&mut Vec<Kf>> {
        match track {
            0 => Some(&mut self.bright_kf),
            1 => Some(&mut self.contrast_kf),
            2 => Some(&mut self.sat_kf),
            3 => Some(&mut self.opacity_kf),
            _ => None,
        }
    }

    /// Move grade keyframe `idx` of `track` to timeline frame `new_t` (clamped to `>= 0`), then
    /// re-sort that track ascending by `t` so eval/draw stay correct (mirrors MojoMedia kf_set's
    /// sorted ordering — here applied to a moved key rather than a fresh one). The key keeps its
    /// VALUE; only its frame changes. No-op for a bad track or `idx`. If the move lands the key
    /// exactly onto another key's frame, BOTH are kept (a stable sort preserves their relative
    /// order) — eval_track still returns a well-defined value, and a later add_grade_key at that
    /// frame would collapse them via set_track's replace.
    pub fn move_grade_key(&mut self, track: u8, idx: usize, new_t: i64) {
        let nt = new_t.max(0);
        if let Some(t) = self.grade_track_mut(track) {
            if idx < t.len() {
                t[idx].t = nt;
                // stable sort by frame so the moved key slots into ascending order
                t.sort_by_key(|k| k.t);
            }
        }
    }

    /// Delete grade keyframe `idx` of `track` (no-op for a bad track or `idx`). Removing a key
    /// keeps the track sorted (Vec::remove preserves order).
    pub fn delete_grade_key(&mut self, track: u8, idx: usize) {
        if let Some(t) = self.grade_track_mut(track) {
            if idx < t.len() {
                t.remove(idx);
            }
        }
    }

    /// Move PiP keyframe `idx` (index into the flat `pip_kf` store) to clip-local frame
    /// `new_t_local` (clamped to `>= 0`). The flat store is unsorted (eval_pip scans it), so no
    /// re-sort is needed; only the entry's `t_local` changes (its clip/par/value are preserved).
    /// No-op for an out-of-range `idx`.
    pub fn move_pip_key(&mut self, idx: usize, new_t_local: i64) {
        if let Some(k) = self.pip_kf.get_mut(idx) {
            k.t_local = new_t_local.max(0);
        }
    }

    /// Delete PiP keyframe `idx` from the flat store (no-op for an out-of-range `idx`).
    pub fn delete_pip_key(&mut self, idx: usize) {
        if idx < self.pip_kf.len() {
            self.pip_kf.remove(idx);
        }
    }

    // ----- per-track state helpers -------------------------------------
    // `track` is the Clip.track index space: 0 = V1, 1 = V2, 2 = A1. Each helper
    // bounds-checks the index (out-of-range tracks are treated as visible / audible /
    // unlocked) so callers never index a 3-element array out of bounds.

    /// True if the given track's VIDEO is hidden (skipped in base/over resolution).
    pub fn is_hidden(&self, track: u8) -> bool {
        (track as usize) < 3 && self.track_hide[track as usize]
    }

    /// True if the given track's AUDIO is muted (contributes nothing to the render).
    pub fn is_muted(&self, track: u8) -> bool {
        (track as usize) < 3 && self.track_mute[track as usize]
    }

    /// True if edits to the given track are blocked (advisory this wave).
    pub fn is_locked(&self, track: u8) -> bool {
        (track as usize) < 3 && self.track_lock[track as usize]
    }

    // ----- edit ops -----------------------------------------------------
    // Mirror MojoMedia editor/main_editor.mojo: positioned clips (explicit t0),
    // split keeps src_in/len/t0 math identical, trims clamp len>=1 and t0>=0.

    /// Append a clip to the timeline.
    pub fn add_clip(&mut self, clip: Clip) {
        self.clips.push(clip);
    }

    /// Remove clip `i` (no-op if out of range). Also keeps the flat PiP keyframe store
    /// clip-stable: drop every PiP key bound to the removed clip, then shift the `clip` index
    /// of keys for higher clips down by one so they keep pointing at the same clip after the
    /// `Vec::remove`. (Grade keys are program-wide and need no remap.)
    pub fn delete_clip(&mut self, i: usize) {
        if i < self.clips.len() {
            self.clips.remove(i);
            self.pip_kf.retain(|k| k.clip != i);
            for k in self.pip_kf.iter_mut() {
                if k.clip > i {
                    k.clip -= 1;
                }
            }
        }
    }

    /// Split clip `i` at timeline frame `t` into two clips. The second clip starts at
    /// `t` with `src_in += off`, `len -= off`, `t0 = t`, same track/look/etc. Mirrors
    /// the razor in main_editor.mojo (`off = cur_T - seg_t0`; split only when 0 < off < len).
    /// Returns the index of the new (right-hand) clip if a split occurred.
    pub fn split_clip(&mut self, i: usize, t: i64) -> Option<usize> {
        if i >= self.clips.len() {
            return None;
        }
        let off = t - self.clips[i].t0;
        if off <= 0 || off >= self.clips[i].len {
            return None; // playhead not strictly inside the clip body
        }
        let mut right = self.clips[i].clone();
        // left half: shorten to the cut point
        self.clips[i].len = off;
        // right half: advance source + start, shorten remaining length
        right.src_in += off;
        right.len -= off;
        right.t0 = self.clips[i].t0 + off;
        // a fresh fade-in on the right half is not inherited from the left's fade-in
        right.fade_in = 0;
        // the left half no longer carries a fade-out (that belongs to the right edge now)
        self.clips[i].fade_out = 0;
        let idx = i + 1;
        self.clips.insert(idx, right);
        // PiP keyframe clip-stability: inserting the right half at `idx` shifts every clip
        // above `i` up by one, so bump the `clip` index of keys for those clips to match. Keys
        // on the split clip itself (clip == i) stay with the LEFT half — their clip-local
        // frames still measure from the same t0, so they animate the (now shorter) left clip.
        // The right half starts with no PiP keys (it inherits the static rect, like a fresh
        // fade-in), mirroring MojoMedia's razor which does not duplicate per-clip keyframes.
        for k in self.pip_kf.iter_mut() {
            if k.clip > i {
                k.clip += 1;
            }
        }
        Some(idx)
    }

    /// Trim the start of clip `i` to a new timeline start `new_t0`, holding the right
    /// edge fixed. Advances src_in by the delta and shortens len, with len>=1 / t0>=0
    /// clamping. Mirrors a left-edge trim (the inverse of the right-edge trim drag).
    pub fn trim_start(&mut self, i: usize, new_t0: i64) {
        if i >= self.clips.len() {
            return;
        }
        let end = self.clips[i].end(); // right edge stays put
        let src_in = self.clips[i].src_in;
        let t0 = self.clips[i].t0;
        let mut nt0 = new_t0.max(0);
        if nt0 >= end {
            nt0 = end - 1; // keep len >= 1
        }
        // Frames trimmed off the head (negative = extend left). On an extend, the source
        // can only supply `src_in` extra frames before hitting frame 0; clamp the timeline
        // extension to that so len/src_in/t0 stay consistent (no out-of-range source request
        // in worker.rs). This mirrors MojoMedia holding the trim within available source.
        let mut delta = nt0 - t0;
        if delta < 0 && src_in + delta < 0 {
            delta = -src_in; // only extend by what the source has
            nt0 = t0 + delta;
        }
        self.clips[i].src_in = (src_in + delta).max(0);
        self.clips[i].t0 = nt0;
        self.clips[i].len = (end - nt0).max(1);
    }

    /// Trim the end of clip `i` to a new length `new_len` (right-edge drag). Clamps to
    /// MIN_CLIP frames (matching MojoMedia's `nl < 15 → 15` trim floor) so dragging the
    /// right edge left past the start — or snapping it to an earlier edge — cannot collapse
    /// the clip below a usable length.
    pub fn trim_end(&mut self, i: usize, new_len: i64) {
        if i >= self.clips.len() {
            return;
        }
        self.clips[i].len = new_len.max(MIN_CLIP);
    }
}

/// Minimum clip length in frames. Mirrors MojoMedia's trim floor (`nl < 15 → 15`).
pub const MIN_CLIP: i64 = 15;

/// Snapshot-based undo/redo for the whole project. `Project` derives `Clone`, so each
/// snapshot is a full copy — simple + correct (mirrors MojoMedia's `Snap` stacks).
///
/// NOTE: not yet wired into `app.rs` (no caller pushes pre-edit state). `#[allow(dead_code)]`
/// keeps the build green under `-D warnings` until the integrator wires undo/redo keybindings
/// + a `push` before each edit. Remove the allow once consumed.
#[allow(dead_code)]
#[derive(Default)]
pub struct History {
    undo: Vec<Project>,
    redo: Vec<Project>,
}

impl History {
    pub fn new() -> History {
        History { undo: Vec::new(), redo: Vec::new() }
    }

    /// Push the *current* project state onto the undo stack before a mutation, clearing
    /// the redo stack (a new edit invalidates the redo future). Call this with the state
    /// as it is *before* applying an edit.
    pub fn push(&mut self, project: &Project) {
        self.undo.push(project.clone());
        self.redo.clear();
    }

    /// Undo: restore the most recent snapshot into `project`, stashing the current state
    /// onto the redo stack. Returns true if anything was undone.
    pub fn undo(&mut self, project: &mut Project) -> bool {
        if let Some(prev) = self.undo.pop() {
            self.redo.push(project.clone());
            *project = prev;
            true
        } else {
            false
        }
    }

    /// Redo: re-apply the most recently undone snapshot, stashing the current state back
    /// onto the undo stack. Returns true if anything was redone.
    pub fn redo(&mut self, project: &mut Project) -> bool {
        if let Some(next) = self.redo.pop() {
            self.undo.push(project.clone());
            *project = next;
            true
        } else {
            false
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grade_keyframe_interp() {
        let mut p = Project::demo("x".into());
        p.bright = 0.0;
        p.add_grade_key(0);
        p.bright = 1.0;
        p.add_grade_key(100);
        assert!((p.grade_at(0).0 - 0.0).abs() < 1e-4);
        assert!((p.grade_at(50).0 - 0.5).abs() < 1e-3, "b50={}", p.grade_at(50).0);
        assert!((p.grade_at(100).0 - 1.0).abs() < 1e-4);
        assert!((p.grade_at(150).0 - 1.0).abs() < 1e-4); // clamp past last key
    }

    #[test]
    fn pip_keyframe_interp() {
        let mut p = Project::demo("x".into());
        p.clips[0].px = 0.0;
        p.add_pip_key(0, 0);
        p.clips[0].px = 0.6;
        p.add_pip_key(0, 100);
        assert!((p.pip_at(0, 0).0 - 0.0).abs() < 1e-4);
        assert!((p.pip_at(0, 50).0 - 0.3).abs() < 1e-3, "px50={}", p.pip_at(0, 50).0);
        assert!((p.pip_at(0, 100).0 - 0.6).abs() < 1e-4);
    }
}
