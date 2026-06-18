//! Project save/load — serde JSON round-trip of the `model::Project`.
//!
//! The Rust win over MojoMedia's line-based text format (editor/project_io.mojo): the
//! whole Project — media paths, names, clips (with all 13 per-clip fields), transitions,
//! the global grade, and markers — serializes/parses with serde for free. We mirror the
//! exact field set MojoMedia persisted (CLIP media/src_in/len/t0/track/look/look_amt/
//! fade_in/fade_out/px/py/pw/ph, TRANS, GRADE bright/contrast/saturation), plus the
//! media library + markers that the Rust model carries.
//!
//! Requires the `serde` (features=["derive"]) + `serde_json` crates — derives live on
//! Clip + Project in model.rs.

use crate::model::Project;

/// Serialize `project` to pretty JSON and write it to `path`.
pub fn save(project: &Project, path: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(project)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

/// Load a `Project` from a JSON file. Returns `None` if the file is missing or the JSON
/// fails to parse (callers fall back to a demo/empty project).
pub fn load(path: &str) -> Option<Project> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}
