//! Project data model — the Rust win over MojoMedia's flat parallel lists: real structs.
//!
//! SHARED CONTRACT. Owned by the timeline/model team; consumed by worker + app + pool.
//! Frame units are timeline frames (30 fps assumed for now).

#[derive(Clone)]
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
pub struct Project {
    pub media: Vec<String>, // media file paths; clips index into this
    pub names: Vec<String>, // display names per media
    pub clips: Vec<Clip>,
    pub trans: Vec<i32>, // transition id per boundary (-1 = none)
    pub bright: f32,
    pub contrast: f32,
    pub sat: f32,
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
        }
    }

    pub fn total_frames(&self) -> i64 {
        self.clips.iter().map(|c| c.end()).max().unwrap_or(1).max(1)
    }
}
