// Copyright Kani Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Function- and harness-scoped loop bounds (`--loop-bounds FILE --loop-bounds-evidence DIR`,
//! `-Z unstable-options`).
//!
//! A policy file names loops by what the source fixes: the function holding the loop (its pretty
//! name, generic arguments included), the source file of that function, the number of loops the
//! function is expected to have, the loop's number within it, the bound, and the reason the bound
//! holds. Each loop's bound is declared once, at the scope its reason belongs to: a bound that is a
//! property of the code (a loop over a fixed table) is function-scoped, applied in every harness
//! whose binary holds the function; a bound that rests on one harness's domain is that harness's.
//! No loop is bounded in both scopes. The concrete CBMC loop identifiers are never part of the
//! policy: they are derived from the goto binary CBMC is about to check, after Kani's
//! instrumentation has rewritten it in place (harness specialization, contract instrumentation,
//! unused-function removal and back-edge rewriting can each change the loops a function holds).
//! Resolution therefore runs on that final binary, immediately before CBMC is invoked on it, and
//! each resolved bound reaches CBMC as `--unwindset ID:BOUND` in the arguments of that harness's
//! run alone; the session's arguments are never modified, so harnesses verified in parallel cannot
//! observe each other's bounds.
//!
//! What a selector establishes, and what it does not: resolution checks that the function exists in
//! the instrumented binary, that it holds exactly the declared number of loops, that every one of
//! them lies in the declared file, that they belong to one function symbol, that the selected loop
//! number is one of them, and that no two selectors resolve to one loop. It does not establish which
//! source loop a number denotes: a compiler or instrumentation change that exchanges the numbers of
//! two loops of one function satisfies every check. The source location CBMC records for the
//! resolved loop is kept in the evidence so such an exchange is visible, and the bound's soundness
//! rests on CBMC's unwinding assertions, which this mode requires (`--no-unwinding-checks` is
//! refused): an execution of the harness's domain that needs more iterations of the loop than its
//! bound fails the harness, whichever loop the number denotes.
//!
//! CBMC itself only warns about an `--unwindset` identifier it does not know and drops the bound
//! (`goto-programs/unwindset.cpp`), which is why resolution is checked here rather than left to it.
//!
//! Evidence: the caller names a directory it owns (`--loop-bounds-evidence`); every harness
//! verified in the run gets a subdirectory of its own, named by its pretty name with `::` written
//! `.`, created exclusively, as is every file in it, so an existing destination is an error and
//! nothing is ever truncated. Every harness's instrumented binary is first copied into its
//! directory (`artifact.out`, recorded with its build path and digest in `artifact.json`), and that
//! copy is the one binary the harness's run reads from then on: the build directory may be
//! rewritten by a later run or another instrumentation of the same path without changing what was
//! checked, and the copy survives a refused resolution or a failed verification for replay. For
//! every harness, the loop discovery's invocation, termination, stdout and stderr are written
//! before they are parsed, and the inventory is validated; for a harness the policy bounds, then
//! the resolution (the harness, its policy entry, the binary's SHA-256 digest and each resolved
//! loop with its source location). An inventory or resolution that is refused leaves its error
//! (`resolution.error`). For every harness, the final CBMC command (NUL-ended
//! parts), the driver and solver executables (canonical path and SHA-256 digest), and the launch
//! attempt are recorded, in files and in the harness's report, immediately before CBMC is spawned:
//! a refused policy leaves its discovery and its error and no attempt.
//!
//! Deadlines: the policy gives every harness the run verifies its own deadline (`deadlines`, the
//! milliseconds and the reason they follow from the harness's domain), and `--harness-timeout` is
//! refused beside it. Every solver execution that reaches spawn records one timing record
//! (`solver.execution.json`): the deadline it was given, the time from immediately after CBMC
//! was spawned to its end (CBMC's output read to its end and its exit reaped, the deadline's
//! detection, or an execution error; compilation, instrumentation, preservation and discovery
//! come before it, rendering after), and how it ended: CBMC's exit code or signal, kept apart;
//! the deadline, with CBMC's last status message and the separate time of its termination and
//! reaping; or the driver's execution error. A run that reaches its deadline has no verdict. A
//! breach shows that the harness is wrongly sized for its expectation, that the expectation is
//! wrong, or that the code it checks regressed. An expectation is never raised solely to turn a
//! failed run green: a correction needs its own justification, the failure is kept, and the run is
//! repeated.

use anyhow::{Context, Result, bail};
use kani_metadata::HarnessMetadata;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The prefix of the line reporting the digest of the binary a harness's bounds were resolved
/// against and CBMC then checked.
pub const ARTIFACT_LINE: &str = "Loop bounds artifact: ";
/// The prefix of each line reporting one resolved bound.
pub const BOUND_LINE: &str = "Loop bound: ";
/// The prefix of the line recording that CBMC is about to be spawned for a harness.
pub const LAUNCH_LINE: &str = "CBMC launch attempt: ";

/// A `--loop-bounds` file: the loops whose bound is a property of the code that holds them, each
/// applied in every harness whose binary holds its function, the harnesses that take loop
/// bounds of their own, and each harness's deadline.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub loops: Vec<Selector>,
    pub harnesses: Vec<HarnessPolicy>,
    pub deadlines: Vec<Deadline>,
}

/// The longest one harness's CBMC run may take, measured from immediately after CBMC is spawned,
/// and why: the caller's margin over the run's supported expectation. A run that reaches it is
/// killed and has no verdict ([`SolverTermination::Deadline`]); a deadline is never evidence of a
/// proof, and an expectation is never raised solely to turn a failed run green.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Deadline {
    pub harness: String,
    pub milliseconds: u64,
    pub reason: String,
}

/// One harness's loop bounds, the harness named by its fully qualified pretty name.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HarnessPolicy {
    pub harness: String,
    pub loops: Vec<Selector>,
}

/// One loop, the bound CBMC unwinds it to, and why that bound covers the harness's domain.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Selector {
    /// The function holding the loop, as CBMC's loop inventory names it (`sourceLocation.function`).
    pub function: String,
    /// The source file of every loop of that function (`sourceLocation.file`).
    pub file: String,
    /// How many loops the function holds in the instrumented binary.
    pub loops_in_function: u32,
    /// The loop's number within the function (the `.N` of its CBMC identifier).
    #[serde(rename = "loop")]
    pub number: u32,
    /// The bound: CBMC unwinds the loop this many times, then asserts it needs no more.
    pub bound: u32,
    /// Why the bound covers every execution of the harness's domain; the unwinding assertion is
    /// what checks it.
    pub reason: String,
}

impl Policy {
    /// Read and validate the policy at `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read the loop-bounds file {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("invalid loop-bounds file {}", path.display()))
    }

    /// Parse and validate a policy: harnesses named once each, every harness with at least one
    /// bound; in each scope, every bound with a reason, no loop selected twice, every selected
    /// loop number below its function's declared loop count, and one loop count and one file per
    /// function; and no loop bounded both by the function scope and by a harness.
    pub fn parse(text: &str) -> Result<Self> {
        let policy: Policy = serde_json::from_str(text)?;
        check_selectors("the function scope", &policy.loops)?;
        let scoped: BTreeSet<(&str, u32)> = policy
            .loops
            .iter()
            .map(|selector| (selector.function.as_str(), selector.number))
            .collect();
        let mut harnesses = BTreeSet::new();
        for entry in &policy.harnesses {
            let scope = format!("harness `{}`", entry.harness);
            if !harnesses.insert(entry.harness.as_str()) {
                bail!("{scope} is named more than once");
            }
            if entry.loops.is_empty() {
                bail!("{scope} declares no loop bound");
            }
            check_selectors(&scope, &entry.loops)?;
            if let Some(both) = entry
                .loops
                .iter()
                .find(|selector| scoped.contains(&(selector.function.as_str(), selector.number)))
            {
                bail!(
                    "{scope}: loop {} of `{}` is bounded by the function scope too",
                    both.number,
                    both.function
                );
            }
        }
        let mut deadlines = BTreeSet::new();
        for deadline in &policy.deadlines {
            let scope = format!("the deadline of harness `{}`", deadline.harness);
            if !deadlines.insert(deadline.harness.as_str()) {
                bail!("{scope} is declared more than once");
            }
            if deadline.milliseconds == 0 {
                bail!("{scope} is zero milliseconds");
            }
            if deadline.reason.trim().is_empty() {
                bail!("{scope} gives no reason");
            }
        }
        Ok(policy)
    }

    /// Refuse a policy naming a harness this run does not verify. The compiler keeps only the
    /// harnesses the run's filters select (`kani_middle::codegen_units::determine_targets`), so the
    /// caller passes the policy of exactly its selection; a name outside it is a renamed, removed
    /// or unselected harness, whose bounds would otherwise be lost without notice.
    pub fn check_harnesses(&self, harnesses: &[&HarnessMetadata]) -> Result<()> {
        let known: BTreeSet<&str> = harnesses.iter().map(|h| h.pretty_name.as_str()).collect();
        let unknown: Vec<&str> = self
            .harnesses
            .iter()
            .map(|entry| entry.harness.as_str())
            .chain(self.deadlines.iter().map(|deadline| deadline.harness.as_str()))
            .filter(|name| !known.contains(name))
            .collect();
        if !unknown.is_empty() {
            bail!("the loop-bounds file names harnesses this run does not verify: {unknown:?}");
        }
        // Every harness verified has its own deadline: none falls back to an unbounded run or to
        // another harness's.
        let undated: Vec<&str> =
            known.iter().copied().filter(|name| self.deadline(name).is_none()).collect();
        if !undated.is_empty() {
            bail!(
                "the loop-bounds file gives no deadline to harnesses this run verifies: {undated:?}"
            );
        }
        Ok(())
    }

    /// The bounds of `harness`, if the policy has any.
    pub fn for_harness(&self, harness: &str) -> Option<&HarnessPolicy> {
        self.harnesses.iter().find(|entry| entry.harness == harness)
    }

    /// The deadline of `harness` ([`Self::check_harnesses`] requires one of every harness a run
    /// verifies).
    pub fn deadline(&self, harness: &str) -> Option<&Deadline> {
        self.deadlines.iter().find(|deadline| deadline.harness == harness)
    }
}

/// The selectors of one `scope`, validated: every bound with a reason, every selected loop number
/// below its function's declared loop count, no loop selected twice, and one loop count and one
/// file per function.
fn check_selectors(scope: &str, selectors: &[Selector]) -> Result<()> {
    let mut selected = BTreeSet::new();
    let mut functions: BTreeMap<&str, (&str, u32)> = BTreeMap::new();
    for selector in selectors {
        if selector.reason.trim().is_empty() {
            bail!(
                "{scope}: the bound of loop {} of `{}` gives no reason",
                selector.number,
                selector.function
            );
        }
        if selector.number >= selector.loops_in_function {
            bail!(
                "{scope}: loop {} of `{}` is not one of its {} loop(s)",
                selector.number,
                selector.function,
                selector.loops_in_function
            );
        }
        if !selected.insert((selector.function.as_str(), selector.number)) {
            bail!(
                "{scope}: loop {} of `{}` is bounded more than once",
                selector.number,
                selector.function
            );
        }
        let shape = (selector.file.as_str(), selector.loops_in_function);
        if let Some(previous) = functions.insert(selector.function.as_str(), shape)
            && previous != shape
        {
            bail!(
                "{scope}: `{}` is declared both as {} loop(s) in {} and as {} loop(s) in {}",
                selector.function,
                previous.1,
                previous.0,
                shape.1,
                shape.0
            );
        }
    }
    Ok(())
}

/// A run's loop bounds: the validated policy, the evidence directory the caller owns, and, per
/// function-scoped selector (by index in [`Policy::loops`]), the harnesses of the run that
/// resolved it.
#[derive(Debug)]
pub struct LoopBounds {
    pub policy: Policy,
    evidence: PathBuf,
    applied: std::sync::Mutex<BTreeMap<usize, BTreeSet<String>>>,
}

impl LoopBounds {
    /// The policy at `policy`, its evidence written under `evidence`, an existing directory.
    pub fn load(policy: &Path, evidence: &Path) -> Result<Self> {
        let policy = Policy::load(policy)?;
        if !evidence.is_dir() {
            bail!("the loop-bounds evidence directory {} does not exist", evidence.display());
        }
        Ok(Self { policy, evidence: evidence.to_path_buf(), applied: Default::default() })
    }

    /// Record, once every harness of the run has been checked, each function-scoped selector
    /// with the harnesses whose binary it bounded (`function-scope.json` in the evidence
    /// directory, created exclusively). A selector no harness of a run resolved is recorded with
    /// none: whether that is a loss is for the caller that knows which harnesses the run covers.
    pub fn record_function_scope(&self) -> Result<()> {
        let applied = self.applied.lock().expect("the applied record is never poisoned");
        let record: Vec<serde_json::Value> = self
            .policy
            .loops
            .iter()
            .enumerate()
            .map(|(index, selector)| {
                serde_json::json!({
                    "selector": selector,
                    "harnesses": applied.get(&index).cloned().unwrap_or_default(),
                })
            })
            .collect();
        HarnessEvidence { directory: self.evidence.clone() }
            .write("function-scope.json", serde_json::to_string_pretty(&record)?.as_bytes())
    }

    /// The evidence of `harness`: its subdirectory, created exclusively.
    pub fn reserve(&self, harness: &HarnessMetadata) -> Result<HarnessEvidence> {
        let directory = self.evidence.join(harness.pretty_name.replace("::", "."));
        std::fs::create_dir(&directory).with_context(|| {
            format!("failed to reserve the loop-bounds evidence {}", directory.display())
        })?;
        Ok(HarnessEvidence { directory })
    }

    /// The loop inventory of `artifact`, the harness's preserved binary
    /// ([`HarnessEvidence::preserve_artifact`]), discovered and validated for every harness, and
    /// the bounds that apply to `harness` resolved against it: its own entry's, and every
    /// function-scoped selector whose function the inventory holds; every step's evidence
    /// written into `evidence` before the next step reads it: the lines reporting the
    /// resolution, then the CBMC arguments applying it. `None` when no bound applies, the
    /// inventory recorded all the same.
    pub fn apply(
        &self,
        harness: &HarnessMetadata,
        artifact: &Artifact,
        evidence: &HarnessEvidence,
    ) -> Result<Option<(Vec<String>, Vec<OsString>)>> {
        let text = evidence.discover(&artifact.path)?;
        let entry = self.policy.for_harness(&harness.pretty_name);
        let resolution = parse_inventory(&text).and_then(|(loops, warnings)| {
            let scoped: Vec<usize> = self
                .policy
                .loops
                .iter()
                .enumerate()
                .filter(|(_, selector)| {
                    loops.iter().any(|listed| {
                        listed.location.function.as_deref() == Some(selector.function.as_str())
                    })
                })
                .map(|(index, _)| index)
                .collect();
            let selectors: Vec<&Selector> = entry
                .into_iter()
                .flat_map(|entry| &entry.loops)
                .chain(scoped.iter().map(|&index| &self.policy.loops[index]))
                .collect();
            let resolved = resolve(&harness.pretty_name, &selectors, &loops)?;
            Ok((warnings, scoped, resolved))
        });
        let (warnings, scoped, resolved) = match resolution {
            Ok(resolution) => resolution,
            Err(error) => {
                evidence.write("resolution.error", format!("{error:#}\n").as_bytes())?;
                return Err(error);
            }
        };
        if resolved.is_empty() {
            return Ok(None);
        }
        let mut applied = self.applied.lock().expect("the applied record is never poisoned");
        for &index in &scoped {
            applied.entry(index).or_default().insert(harness.pretty_name.clone());
        }
        drop(applied);
        let mut lines = Vec::with_capacity(resolved.len() + warnings.len() + 1);
        lines.extend(warnings.iter().map(|warning| format!("Loop inventory warning: {warning}")));
        lines.push(format!(
            "{ARTIFACT_LINE}{} sha256:{}",
            artifact.path.display(),
            artifact.sha256
        ));
        lines.extend(resolved.iter().map(Resolved::render));
        let record = serde_json::json!({
            "harness": harness.pretty_name,
            "policy": entry,
            "function_scope": scoped.iter().map(|&index| &self.policy.loops[index]).collect::<Vec<_>>(),
            "artifact": { "path": artifact.path, "sha256": artifact.sha256 },
            "warnings": warnings,
            "resolved": resolved.iter().map(|entry| serde_json::json!({
                "selector": entry.selector,
                "loop": entry.listed.name,
                "location": entry.listed.location,
            })).collect::<Vec<_>>(),
        });
        evidence.write("resolution.json", serde_json::to_string_pretty(&record)?.as_bytes())?;
        Ok(Some((lines, unwindset(&resolved))))
    }
}

/// A duration as evidence: whole seconds and the nanoseconds below one second, integers, so the
/// record carries the driver's measurement exactly; no clock instant is recorded, since a
/// monotonic instant means nothing outside the process that took it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Elapsed {
    pub seconds: u64,
    pub nanoseconds: u32,
}

impl From<std::time::Duration> for Elapsed {
    fn from(duration: std::time::Duration) -> Self {
        Self { seconds: duration.as_secs(), nanoseconds: duration.subsec_nanos() }
    }
}

/// How one solver execution ended ([`HarnessEvidence::execution`]): CBMC's own exit or signal;
/// killed at its deadline, with CBMC's last status message, the time from the deadline's
/// detection through CBMC's termination and reaping, and the error of that termination if it
/// failed; or an execution error of the driver's, with the operation that failed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SolverTermination {
    Exited {
        code: i32,
    },
    Signaled {
        signal: i32,
    },
    Deadline {
        last_status_message: Option<String>,
        cleanup_elapsed: Elapsed,
        cleanup_error: Option<String>,
    },
    ExecutionError {
        operation: String,
        diagnostic: String,
    },
}

/// One harness's evidence directory ([`LoopBounds::reserve`]).
#[derive(Debug)]
pub struct HarnessEvidence {
    directory: PathBuf,
}

/// A harness's instrumented binary as its evidence holds it: the preserved copy every step of the
/// harness's run reads, and that copy's SHA-256 digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
}

impl HarnessEvidence {
    /// The instrumented binary at `source`, copied byte for byte into this directory's
    /// `artifact.out`, created exclusively, flushed and synchronized, and recorded with its
    /// origin in `artifact.json` (`source`, the build path it was copied from; `path`; `sha256`,
    /// the copy's digest). The copy is what discovery, resolution and the solver then read, so a
    /// later rewrite of the build directory, by another run or another instrumentation of the same
    /// path, cannot change what this harness was checked on, and the copy outlives the run
    /// whatever its outcome.
    pub fn preserve_artifact(&self, source: &Path) -> Result<Artifact> {
        let path = self.directory.join("artifact.out");
        let mut preserved = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        let mut original = std::fs::File::open(source).with_context(|| {
            format!("failed to open the instrumented binary {}", source.display())
        })?;
        std::io::copy(&mut original, &mut preserved)
            .and_then(|_| preserved.flush())
            .and_then(|()| preserved.sync_all())
            .with_context(|| {
                format!("failed to preserve {} as {}", source.display(), path.display())
            })?;
        let artifact = Artifact { sha256: digest(&path)?, path };
        let record = serde_json::json!({
            "source": source,
            "path": artifact.path,
            "sha256": artifact.sha256,
        });
        self.write("artifact.json", serde_json::to_string_pretty(&record)?.as_bytes())?;
        Ok(artifact)
    }

    /// Write `bytes` to `name` in this directory, refusing an existing file.
    pub fn write(&self, name: &str, bytes: &[u8]) -> Result<()> {
        let path = self.directory.join(name);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .with_context(|| format!("failed to write {}", path.display()))
    }

    /// Record the harness's one solver execution (`solver.execution.json`, created exclusively):
    /// the deadline the run was given, the time from immediately after CBMC was spawned to its
    /// end (normal completion, the deadline's detection, or an execution error), and how it ended.
    pub fn execution(
        &self,
        deadline_milliseconds: u64,
        elapsed: std::time::Duration,
        termination: &SolverTermination,
    ) -> Result<()> {
        let record = serde_json::json!({
            "deadline_milliseconds": deadline_milliseconds,
            "elapsed": Elapsed::from(elapsed),
            "termination": termination,
        });
        self.write("solver.execution.json", serde_json::to_string_pretty(&record)?.as_bytes())
    }

    /// Record the command CBMC is about to be spawned as, its program and arguments read from the
    /// command itself, and the attempt to spawn it.
    pub fn launch(&self, program: &OsStr, args: &[OsString]) -> Result<()> {
        self.write("solver.arguments", &invocation(program, args))?;
        // The executables of the run: this driver, and the solver the command spawns, each by its
        // canonical path and SHA-256 digest.
        let driver = std::env::current_exe()
            .and_then(std::fs::canonicalize)
            .context("Cannot determine the driver's executable")?;
        let solver = Path::new(program);
        let identity = serde_json::json!({
            "driver": { "path": driver, "sha256": digest(&driver)? },
            "solver": { "path": solver, "sha256": digest(solver)? },
        });
        self.write("solver.identity", serde_json::to_string_pretty(&identity)?.as_bytes())?;
        self.write("solver.attempt", b"spawning\n")
    }

    /// The loop inventory of the binary at `file`: `goto-instrument --show-loops --json-ui`, its
    /// invocation, termination, stdout and stderr written before anything reads them; its stdout
    /// as text when it exited successfully.
    fn discover(&self, file: &Path) -> Result<String> {
        let args: Vec<OsString> =
            vec!["--show-loops".into(), "--json-ui".into(), file.as_os_str().to_owned()];
        // The `goto-instrument` on `PATH`, Kani's own, as every other goto-instrument call of the
        // driver.
        let program = "goto-instrument";
        let mut command = Command::new(program);
        command.args(&args);
        let recorded: Vec<OsString> = command.get_args().map(OsStr::to_os_string).collect();
        self.write("discovery.arguments", &invocation(command.get_program(), &recorded))?;
        let output = command.output().with_context(|| format!("Failed to invoke {program}"))?;
        self.write("discovery.status", format!("{}\n", output.status).as_bytes())?;
        self.write("discovery.stdout", &output.stdout)?;
        self.write("discovery.stderr", &output.stderr)?;
        if !output.status.success() {
            bail!(
                "{program} --show-loops {} exited with status {} (its output is in {})",
                file.display(),
                output.status,
                self.directory.display()
            );
        }
        String::from_utf8(output.stdout).context("the loop inventory is not UTF-8")
    }
}

/// An invocation as evidence: the program, then each argument, each ended by a NUL byte, which no
/// process argument contains, so the record splits back into exactly the invocation's parts.
fn invocation(program: &OsStr, args: &[OsString]) -> Vec<u8> {
    let mut text = Vec::new();
    for part in std::iter::once(program).chain(args.iter().map(OsString::as_os_str)) {
        text.extend_from_slice(part.as_encoded_bytes());
        text.push(0);
    }
    text
}

/// One record of `goto-instrument --show-loops --json-ui` (CBMC `goto-programs/loop_ids.cpp`,
/// `show_loop_ids_json`, framed by `ui_message_handlert`): the program banner, the loop inventory,
/// or a message.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Record {
    Loops(LoopsRecord),
    Message(MessageRecord),
    // Read only to recognize the record; its content names the tool.
    Program(#[allow(dead_code)] ProgramRecord),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LoopsRecord {
    loops: Vec<Loop>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MessageRecord {
    #[serde(rename = "messageType")]
    message_type: String,
    #[serde(rename = "messageText")]
    message_text: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProgramRecord {
    #[allow(dead_code)]
    program: String,
}

/// One loop of the inventory: its CBMC identifier and where CBMC locates it.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Loop {
    pub name: String,
    #[serde(rename = "sourceLocation")]
    pub location: Location,
}

/// A loop's source location, as CBMC's `json(source_locationt)` renders it: every field a string.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Location {
    pub function: Option<String>,
    pub file: Option<String>,
    pub line: Option<String>,
    pub column: Option<String>,
}

/// The loops of a `--show-loops --json-ui` document, with the warnings it carried. Refused when a
/// record is an error, when the document holds no inventory or more than one, when a loop
/// identifier is repeated or has no numeric loop number, and when a loop has no function or file.
pub fn parse_inventory(text: &str) -> Result<(Vec<Loop>, Vec<String>)> {
    let records: Vec<Record> = serde_json::from_str(text)
        .context("the loop inventory is not a goto-instrument JSON UI document")?;
    let mut inventories = Vec::new();
    let mut warnings = Vec::new();
    for record in records {
        match record {
            Record::Loops(LoopsRecord { loops }) => inventories.push(loops),
            Record::Message(MessageRecord { message_type, message_text }) => {
                match message_type.as_str() {
                    "ERROR" => bail!("goto-instrument reported an error: {message_text}"),
                    "WARNING" => warnings.push(message_text),
                    _ => {}
                }
            }
            Record::Program(_) => {}
        }
    }
    let loops = match <[Vec<Loop>; 1]>::try_from(inventories) {
        Ok([loops]) => loops,
        Err(inventories) => {
            bail!("the document holds {} loop inventories, not one", inventories.len())
        }
    };
    let mut names = BTreeSet::new();
    for listed in &loops {
        if !names.insert(listed.name.as_str()) {
            bail!("loop `{}` is listed more than once", listed.name);
        }
        loop_number(&listed.name)?;
        if listed.location.function.is_none() || listed.location.file.is_none() {
            bail!("loop `{}` has no function or no file in its source location", listed.name);
        }
    }
    Ok((loops, warnings))
}

/// The function symbol and loop number of a CBMC loop identifier (`SYMBOL.N`).
fn loop_number(name: &str) -> Result<(&str, u32)> {
    name.rsplit_once('.')
        .and_then(|(symbol, number)| Some((symbol, number.parse().ok()?)))
        .filter(|(symbol, _)| !symbol.is_empty())
        .with_context(|| format!("loop identifier `{name}` names no loop number"))
}

/// One selector resolved against a binary's loops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub selector: Selector,
    pub listed: Loop,
}

impl Resolved {
    /// The line reporting this resolution ([`BOUND_LINE`]).
    pub fn render(&self) -> String {
        let location = &self.listed.location;
        let part = |value: &Option<String>| value.clone().unwrap_or_else(|| "?".to_string());
        format!(
            "{BOUND_LINE}{} loop {} of {} at {}:{}:{} = {} unwound to {}",
            self.selector.function,
            self.selector.number,
            self.selector.loops_in_function,
            part(&location.file),
            part(&location.line),
            part(&location.column),
            self.listed.name,
            self.selector.bound
        )
    }
}

/// Resolve each of `selectors`, the bounds that apply to `harness`, against `loops`, the inventory
/// of the binary CBMC is about to check: the function's loops must number exactly as declared,
/// all lie in the declared file, and come from one function symbol; the selected number must be
/// one of them; and no two selectors may resolve to one loop. These checks do not establish which
/// source loop a number denotes (see the module documentation).
pub fn resolve(harness: &str, selectors: &[&Selector], loops: &[Loop]) -> Result<Vec<Resolved>> {
    let mut resolved = Vec::with_capacity(selectors.len());
    let mut used = BTreeSet::new();
    for &selector in selectors {
        let candidates: Vec<&Loop> = loops
            .iter()
            .filter(|listed| {
                listed.location.function.as_deref() == Some(selector.function.as_str())
            })
            .collect();
        let describe = |candidates: &[&Loop]| {
            candidates.iter().fold(String::new(), |mut text, listed| {
                let location = &listed.location;
                let _ = write!(
                    text,
                    "\n  {} at {}:{}",
                    listed.name,
                    location.file.as_deref().unwrap_or("?"),
                    location.line.as_deref().unwrap_or("?")
                );
                text
            })
        };
        if candidates.len() != selector.loops_in_function as usize {
            bail!(
                "harness `{}`: `{}` has {} loop(s) in the instrumented binary, the policy declares \
                 {}:{}",
                harness,
                selector.function,
                candidates.len(),
                selector.loops_in_function,
                describe(&candidates)
            );
        }
        if let Some(elsewhere) = candidates
            .iter()
            .find(|listed| listed.location.file.as_deref() != Some(selector.file.as_str()))
        {
            bail!(
                "harness `{}`: loop `{}` of `{}` is in {}, the policy declares {}",
                harness,
                elsewhere.name,
                selector.function,
                elsewhere.location.file.as_deref().unwrap_or("?"),
                selector.file
            );
        }
        let mut symbols = BTreeSet::new();
        let mut chosen = Vec::new();
        for listed in &candidates {
            let (symbol, number) = loop_number(&listed.name)?;
            symbols.insert(symbol);
            if number == selector.number {
                chosen.push(*listed);
            }
        }
        if symbols.len() != 1 {
            bail!(
                "harness `{}`: `{}` names {} function symbols in the instrumented binary:{}",
                harness,
                selector.function,
                symbols.len(),
                describe(&candidates)
            );
        }
        let listed = match chosen.as_slice() {
            [one] => (*one).clone(),
            _ => bail!(
                "harness `{}`: `{}` has no loop {} in the instrumented binary:{}",
                harness,
                selector.function,
                selector.number,
                describe(&candidates)
            ),
        };
        if !used.insert(listed.name.clone()) {
            bail!("harness `{harness}`: two bounds resolve to loop `{}`", listed.name);
        }
        resolved.push(Resolved { selector: selector.clone(), listed });
    }
    Ok(resolved)
}

/// The CBMC arguments applying `resolved`: one `--unwindset` of every `ID:BOUND`.
pub fn unwindset(resolved: &[Resolved]) -> Vec<OsString> {
    let set = resolved
        .iter()
        .map(|entry| format!("{}:{}", entry.listed.name, entry.selector.bound))
        .collect::<Vec<_>>()
        .join(",");
    vec!["--unwindset".into(), set.into()]
}

/// The lowercase hexadecimal SHA-256 digest of the file at `path`: a resolved goto binary, the
/// driver, the solver.
pub fn digest(path: &Path) -> Result<String> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(Sha256::digest(&bytes).iter().fold(String::with_capacity(64), |mut text, byte| {
        let _ = write!(text, "{byte:02x}");
        text
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A document in the layout `goto-instrument --show-loops --json-ui` of CBMC 6.11.0 (Kani
    /// 0.68.0's) prints: the program banner, the inventory, then the status message, every
    /// location field a string. The loop entries are synthetic, a two-function program's; the
    /// real listing of a real instrumented binary is exercised end to end by the checks of the
    /// installed driver.
    const INVENTORY: &str = r#"[
  {
    "program": "goto-instrument"
  },
  {
    "loops": [
      {
        "name": "_RNvCs1_4demo8count_up.0",
        "sourceLocation": {
          "column": "5",
          "file": "demo.rs",
          "function": "count_up",
          "line": "3"
        }
      },
      {
        "name": "_RNvCs1_4demo4pair.0",
        "sourceLocation": {
          "column": "5",
          "file": "demo.rs",
          "function": "pair",
          "line": "9"
        }
      },
      {
        "name": "_RNvCs1_4demo4pair.1",
        "sourceLocation": {
          "column": "9",
          "file": "demo.rs",
          "function": "pair",
          "line": "10"
        }
      }
    ]
  },
  {
    "messageText": "Reading GOTO program from 'demo.out'",
    "messageType": "STATUS-MESSAGE"
  }
]"#;

    fn selector(function: &str, loops_in_function: u32, number: u32, bound: u32) -> Selector {
        Selector {
            function: function.to_string(),
            file: "demo.rs".to_string(),
            loops_in_function,
            number,
            bound,
            reason: "the test's".to_string(),
        }
    }

    /// `selectors` resolved for the harness `h` against `loops`.
    fn resolved(selectors: &[Selector], loops: &[Loop]) -> Result<Vec<Resolved>> {
        resolve("h", &selectors.iter().collect::<Vec<_>>(), loops)
    }

    #[test]
    fn an_inventory_reads_its_loops() {
        let (loops, warnings) = parse_inventory(INVENTORY).unwrap();
        assert_eq!(loops.len(), 3);
        assert!(warnings.is_empty());
        assert_eq!(loops[1].location.line.as_deref(), Some("9"));
    }

    #[test]
    fn a_selector_resolves_to_its_loop_and_bound() {
        let (loops, _) = parse_inventory(INVENTORY).unwrap();
        let resolved =
            resolved(&[selector("pair", 2, 1, 4), selector("count_up", 1, 0, 0)], &loops).unwrap();
        assert_eq!(resolved[0].listed.name, "_RNvCs1_4demo4pair.1");
        assert_eq!(
            unwindset(&resolved),
            [
                OsString::from("--unwindset"),
                OsString::from("_RNvCs1_4demo4pair.1:4,_RNvCs1_4demo8count_up.0:0")
            ]
        );
        assert_eq!(
            resolved[0].render(),
            "Loop bound: pair loop 1 of 2 at demo.rs:10:9 = _RNvCs1_4demo4pair.1 unwound to 4"
        );
    }

    #[test]
    fn a_selector_without_its_loop_is_refused() {
        let (loops, _) = parse_inventory(INVENTORY).unwrap();
        for (selectors, expected) in [
            ([selector("absent", 1, 0, 1)], "has 0 loop(s)"),
            ([selector("pair", 3, 2, 1)], "has 2 loop(s)"),
            ([selector("count_up", 2, 1, 1)], "has 1 loop(s)"),
        ] {
            let error = resolved(&selectors, &loops).unwrap_err().to_string();
            assert!(error.contains(expected), "{error}");
        }
        let mut elsewhere = selector("pair", 2, 0, 1);
        elsewhere.file = "other.rs".to_string();
        let error = resolved(&[elsewhere], &loops).unwrap_err().to_string();
        assert!(error.contains("is in demo.rs"), "{error}");
    }

    #[test]
    fn two_symbols_for_one_function_are_ambiguous() {
        let text = INVENTORY.replace("_RNvCs1_4demo4pair.1", "_RNvCs2_4demo4pair.1");
        let (loops, _) = parse_inventory(&text).unwrap();
        let error = resolved(&[selector("pair", 2, 0, 1)], &loops).unwrap_err();
        assert!(error.to_string().contains("names 2 function symbols"), "{error}");
    }

    #[test]
    fn a_malformed_inventory_is_refused() {
        let no_inventory = r#"[{"program": "goto-instrument"}]"#;
        let two = format!("[{}, {}]", r#"{"loops": []}"#, r#"{"loops": []}"#);
        let repeated = INVENTORY.replace("_RNvCs1_4demo4pair.1", "_RNvCs1_4demo4pair.0");
        let unnumbered = INVENTORY.replace("_RNvCs1_4demo4pair.1", "_RNvCs1_4demo4pair.x");
        let located = INVENTORY.replace(r#""function": "pair","#, "");
        let error = r#"[{"messageType": "ERROR", "messageText": "failed to read"}]"#;
        let unknown = r#"[{"loops": [], "extra": 1}]"#;
        for (text, expected) in [
            (no_inventory, "0 loop inventories"),
            (two.as_str(), "2 loop inventories"),
            (repeated.as_str(), "listed more than once"),
            (unnumbered.as_str(), "names no loop number"),
            (located.as_str(), "no function or no file"),
            (error, "reported an error: failed to read"),
            (unknown, "not a goto-instrument JSON UI document"),
            ("not json", "not a goto-instrument JSON UI document"),
        ] {
            let found = parse_inventory(text).unwrap_err().to_string();
            assert!(found.contains(expected), "{expected}: {found}");
        }
        let warned = INVENTORY.replace("STATUS-MESSAGE", "WARNING");
        assert_eq!(parse_inventory(&warned).unwrap().1, ["Reading GOTO program from 'demo.out'"]);
    }

    #[test]
    fn a_policy_is_validated_before_use() {
        let valid = r#"{"loops": [
            {"function": "count_up", "file": "demo.rs", "loops_in_function": 1, "loop": 0,
             "bound": 0, "reason": "the code's"}
        ], "harnesses": [{"harness": "h", "loops": [
            {"function": "pair", "file": "demo.rs", "loops_in_function": 2, "loop": 1, "bound": 4,
             "reason": "the test's"}
        ]}], "deadlines": [{"harness": "h", "milliseconds": 2000, "reason": "twice 1 s"}]}"#;
        let policy = Policy::parse(valid).unwrap();
        assert_eq!(policy.harnesses[0].loops[0], selector("pair", 2, 1, 4));
        assert_eq!(policy.loops[0].function, "count_up");
        assert_eq!(policy.deadline("h").unwrap().milliseconds, 2000);
        assert!(policy.deadline("g").is_none());
        let entry = |loops: &str| format!(r#"{{"harness": "h", "loops": [{loops}]}}"#);
        let bound = |function: &str, file: &str, count: u32, number: u32| {
            format!(
                r#"{{"function": "{function}", "file": "{file}", "loops_in_function": {count}, "loop": {number}, "bound": 1, "reason": "r"}}"#
            )
        };
        let harnesses = |entries: &str| {
            format!(r#"{{"loops": [], "harnesses": [{entries}], "deadlines": []}}"#)
        };
        let deadlines = |entries: &str| {
            format!(r#"{{"loops": [], "harnesses": [], "deadlines": [{entries}]}}"#)
        };
        for (text, expected) in [
            (
                harnesses(&format!(
                    "{}, {}",
                    entry(&bound("f", "a.rs", 1, 0)),
                    entry(&bound("f", "a.rs", 1, 0))
                )),
                "named more than once",
            ),
            (harnesses(&entry("")), "declares no loop bound"),
            (harnesses(&entry(&bound("f", "a.rs", 1, 1))), "is not one of its 1 loop(s)"),
            (
                harnesses(&entry(&format!(
                    "{}, {}",
                    bound("f", "a.rs", 2, 0),
                    bound("f", "a.rs", 2, 0)
                ))),
                "bounded more than once",
            ),
            (
                harnesses(&entry(&format!(
                    "{}, {}",
                    bound("f", "a.rs", 2, 0),
                    bound("f", "b.rs", 2, 1)
                ))),
                "is declared both as",
            ),
            (
                harnesses(&entry(
                    &bound("f", "a.rs", 1, 0).replace(r#""reason": "r""#, r#""reason": " ""#),
                )),
                "gives no reason",
            ),
            (
                format!(
                    r#"{{"loops": [{}, {}], "harnesses": [], "deadlines": []}}"#,
                    bound("f", "a.rs", 1, 0),
                    bound("f", "a.rs", 1, 0)
                ),
                "the function scope: loop 0 of `f` is bounded more than once",
            ),
            (
                format!(
                    r#"{{"loops": [{}], "harnesses": [{}], "deadlines": []}}"#,
                    bound("f", "a.rs", 1, 0),
                    entry(&bound("f", "a.rs", 1, 0))
                ),
                "is bounded by the function scope too",
            ),
            (
                deadlines(
                    r#"{"harness": "h", "milliseconds": 1, "reason": "r"},
                       {"harness": "h", "milliseconds": 2, "reason": "r"}"#,
                ),
                "the deadline of harness `h` is declared more than once",
            ),
            (
                deadlines(r#"{"harness": "h", "milliseconds": 0, "reason": "r"}"#),
                "the deadline of harness `h` is zero milliseconds",
            ),
            (
                deadlines(r#"{"harness": "h", "milliseconds": 1, "reason": " "}"#),
                "the deadline of harness `h` gives no reason",
            ),
            (r#"{"harnesses": [], "deadlines": []}"#.to_string(), "missing field `loops`"),
            (r#"{"loops": [], "harnesses": []}"#.to_string(), "missing field `deadlines`"),
            (
                r#"{"loops": [], "harnesses": [], "deadlines": [], "unknown": 1}"#.to_string(),
                "unknown field",
            ),
        ] {
            let found = format!("{:#}", Policy::parse(&text).unwrap_err());
            assert!(found.contains(expected), "{expected}: {found}");
        }
    }

    /// Every harness a run verifies has its own deadline, and every deadline names one of them:
    /// a harness without one, or a deadline of a harness the run does not verify, is refused.
    #[test]
    fn every_verified_harness_has_its_deadline() {
        let policy = |names: &[&str]| {
            let deadlines: Vec<String> = names
                .iter()
                .map(|name| format!(r#"{{"harness": "{name}", "milliseconds": 1, "reason": "r"}}"#))
                .collect();
            Policy::parse(&format!(
                r#"{{"loops": [], "harnesses": [], "deadlines": [{}]}}"#,
                deadlines.join(", ")
            ))
            .unwrap()
        };
        let a = crate::metadata::tests::mock_proof_harness("a", None, None, None);
        let b = crate::metadata::tests::mock_proof_harness("b", None, None, None);
        policy(&["a", "b"]).check_harnesses(&[&a, &b]).unwrap();
        let undated = policy(&["a"]).check_harnesses(&[&a, &b]).unwrap_err().to_string();
        assert!(
            undated.contains(r#"gives no deadline to harnesses this run verifies: ["b"]"#),
            "{undated}"
        );
        let unknown = policy(&["a", "c"]).check_harnesses(&[&a]).unwrap_err().to_string();
        assert!(
            unknown.contains(r#"names harnesses this run does not verify: ["c"]"#),
            "{unknown}"
        );
    }

    /// The execution record holds the deadline, the exact elapsed duration as integer seconds and
    /// nanoseconds, and the tagged termination; it is created once, and a second record is
    /// refused with the first left as it was.
    #[test]
    fn an_execution_is_recorded_once_exactly() {
        let root = tempfile::tempdir().unwrap();
        let evidence = HarnessEvidence { directory: root.path().to_path_buf() };
        let termination = SolverTermination::Deadline {
            last_status_message: Some("Running Symex".to_string()),
            cleanup_elapsed: Elapsed { seconds: 0, nanoseconds: 2_000_001 },
            cleanup_error: None,
        };
        evidence.execution(2000, std::time::Duration::new(2, 123_456_789), &termination).unwrap();
        let path = root.path().join("solver.execution.json");
        let first = std::fs::read(&path).unwrap();
        let record: serde_json::Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(
            record,
            serde_json::json!({
                "deadline_milliseconds": 2000,
                "elapsed": {"seconds": 2, "nanoseconds": 123_456_789},
                "termination": {
                    "kind": "deadline",
                    "last_status_message": "Running Symex",
                    "cleanup_elapsed": {"seconds": 0, "nanoseconds": 2_000_001},
                    "cleanup_error": null,
                },
            })
        );
        let again = evidence.execution(
            1,
            std::time::Duration::ZERO,
            &SolverTermination::Exited { code: 0 },
        );
        assert!(format!("{:#}", again.unwrap_err()).starts_with("failed to create"));
        assert_eq!(std::fs::read(&path).unwrap(), first);
        for (termination, json) in [
            (
                SolverTermination::Exited { code: 137 },
                serde_json::json!({"kind": "exited", "code": 137}),
            ),
            (
                SolverTermination::Signaled { signal: 9 },
                serde_json::json!({"kind": "signaled", "signal": 9}),
            ),
            (
                SolverTermination::ExecutionError {
                    operation: "o".to_string(),
                    diagnostic: "d".to_string(),
                },
                serde_json::json!({"kind": "execution_error", "operation": "o", "diagnostic": "d"}),
            ),
        ] {
            assert_eq!(serde_json::to_value(&termination).unwrap(), json);
        }
    }

    /// An instrumented binary is preserved once, byte for byte: a later change to its build path
    /// leaves the copy, and the digest every later step names, unchanged; an existing
    /// `artifact.out` is refused and left as it was.
    #[test]
    fn an_artifact_is_preserved_once_and_survives_its_source() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("harness.out");
        std::fs::write(&source, b"instrumented").unwrap();
        let evidence = HarnessEvidence { directory: root.path().join("h") };
        std::fs::create_dir(&evidence.directory).unwrap();
        let artifact = evidence.preserve_artifact(&source).unwrap();
        assert_eq!(artifact.path, evidence.directory.join("artifact.out"));
        assert_eq!(artifact.sha256, digest(&source).unwrap());
        std::fs::write(&source, b"rewritten by a later run").unwrap();
        assert_eq!(std::fs::read(&artifact.path).unwrap(), b"instrumented");
        assert_eq!(digest(&artifact.path).unwrap(), artifact.sha256);
        let record: serde_json::Value = serde_json::from_slice(
            &std::fs::read(evidence.directory.join("artifact.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(record["source"], source.display().to_string());
        assert_eq!(record["path"], artifact.path.display().to_string());
        assert_eq!(record["sha256"], artifact.sha256);
        // The second preservation is refused by `artifact.out`'s own exclusive creation, before
        // anything is copied: not by a later record's.
        let error = format!("{:#}", evidence.preserve_artifact(&source).unwrap_err());
        let refused = format!("failed to create {}", artifact.path.display());
        assert!(error.starts_with(&refused), "{error}");
        assert_eq!(std::fs::read(&artifact.path).unwrap(), b"instrumented");
        // A binary that cannot be read is refused: no step can run on a copy that is not one.
        let missing = HarnessEvidence { directory: root.path().join("missing") };
        std::fs::create_dir(&missing.directory).unwrap();
        let error = missing.preserve_artifact(&root.path().join("absent.out")).unwrap_err();
        assert!(
            format!("{error:#}").contains("failed to open the instrumented binary"),
            "{error:#}"
        );
        assert!(!missing.directory.join("artifact.json").exists());
    }

    /// The evidence of a refused resolution: the discovery's streams and the error are kept,
    /// nothing is overwritten, and a second reservation of the same harness is refused.
    #[test]
    fn evidence_is_written_exclusively_and_survives_refusal() {
        let root = tempfile::tempdir().unwrap();
        let evidence = HarnessEvidence { directory: root.path().join("h") };
        std::fs::create_dir(&evidence.directory).unwrap();
        evidence.write("discovery.stdout", b"not json").unwrap();
        let error = evidence.write("discovery.stdout", b"again").unwrap_err();
        assert!(format!("{error:#}").contains("failed to create"), "{error:#}");
        assert_eq!(
            std::fs::read(evidence.directory.join("discovery.stdout")).unwrap(),
            b"not json"
        );
        // A launch records the command byte for byte, an argument holding a newline included,
        // and the executables that run, by path and digest.
        let solver = std::env::current_exe().unwrap().canonicalize().unwrap();
        evidence
            .launch(
                solver.as_os_str(),
                &["--unwindset".into(), "f.0:1".into(), "two\nlines".into()],
            )
            .unwrap();
        let mut expected = solver.as_os_str().as_encoded_bytes().to_vec();
        expected.extend_from_slice(b"\0--unwindset\0f.0:1\0two\nlines\0");
        assert_eq!(std::fs::read(evidence.directory.join("solver.arguments")).unwrap(), expected);
        let identity: serde_json::Value = serde_json::from_slice(
            &std::fs::read(evidence.directory.join("solver.identity")).unwrap(),
        )
        .unwrap();
        assert_eq!(identity["solver"]["path"], solver.display().to_string());
        assert_eq!(identity["solver"]["sha256"], digest(&solver).unwrap());
        assert_eq!(identity["driver"]["path"], solver.display().to_string());
        assert_eq!(
            std::fs::read(evidence.directory.join("solver.attempt")).unwrap(),
            b"spawning\n"
        );
        assert!(
            evidence.launch(solver.as_os_str(), &[]).is_err(),
            "a second launch record was accepted"
        );
        // A launch whose solver cannot be identified records no attempt: the run stops before
        // the spawn.
        let unknown = HarnessEvidence { directory: root.path().join("unknown") };
        std::fs::create_dir(&unknown.directory).unwrap();
        assert!(unknown.launch(OsStr::new("/nonexistent/cbmc"), &[]).is_err());
        assert!(!unknown.directory.join("solver.attempt").exists());
    }
}
