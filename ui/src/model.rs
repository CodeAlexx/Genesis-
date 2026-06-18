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
}

impl Clip {
    pub fn video(media: usize, t0: i64, len: i64, track: u8, name_hint: &str) -> Clip {
        let _ = name_hint;
        Clip { media, src_in: 0, len, t0, track, look: 0, look_amt: 1.0, fade_in: 0, fade_out: 0, px: 0.0, py: 0.0, pw: 1.0, ph: 1.0 }
    }
    pub fn end(&self) -> i64 {
        self.t0 + self.len
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
        }
    }

    pub fn total_frames(&self) -> i64 {
        self.clips.iter().map(|c| c.end()).max().unwrap_or(1).max(1)
    }

    // ----- edit ops -----------------------------------------------------
    // Mirror MojoMedia editor/main_editor.mojo: positioned clips (explicit t0),
    // split keeps src_in/len/t0 math identical, trims clamp len>=1 and t0>=0.

    /// Append a clip to the timeline.
    pub fn add_clip(&mut self, clip: Clip) {
        self.clips.push(clip);
    }

    /// Remove clip `i` (no-op if out of range).
    pub fn delete_clip(&mut self, i: usize) {
        if i < self.clips.len() {
            self.clips.remove(i);
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
