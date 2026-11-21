// Copyright Kani Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use anyhow::{Result, bail};
use kani_metadata::{CbmcSolver, HarnessMetadata};
use regex::Regex;
use rustc_demangle::demangle;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::ffi::{OsStr, OsString};
use std::fmt::Write;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;
use strum_macros::Display;
use tokio::process::Command as TokioCommand;

use crate::args::common::Verbosity;
use crate::args::{OutputFormat, VerificationArgs};
use crate::cbmc_output_parser::{
    CheckStatus, ParserItem, ProcessTermination, Property, VerificationOutput, extract_results,
    process_cbmc_output,
};
use crate::cbmc_property_renderer::{format_coverage, format_result, kani_cbmc_output_filter};
use crate::coverage::cov_results::{CoverageCheck, CoverageResults};
use crate::coverage::cov_results::{CoverageRegion, CoverageTerm};
use crate::loop_bounds::SolverTermination;
use crate::session::KaniSession;
use crate::util::render_command;

/// CBMC version and system information
#[derive(Debug, Clone)]
pub struct CbmcInfo {
    pub version: String,
    pub os_info: String,
}

/// CBMC runtime and execution statistics
#[derive(Debug, Clone, Default)]
pub struct CbmcStats {
    pub runtime_symex_s: Option<f64>,
    pub size_program_expression: Option<u32>,
    pub slicing_removed_assignments: Option<u32>,
    pub vccs_generated: Option<u32>,
    pub vccs_remaining: Option<u32>,
    pub runtime_postprocess_equation_s: Option<f64>,
    pub runtime_convert_ssa_s: Option<f64>,
    pub runtime_post_process_s: Option<f64>,
    pub runtime_solver_s: Option<f64>,
    pub runtime_decision_procedure_s: Option<f64>,
}

impl KaniSession {
    /// Get CBMC version and system information
    pub fn get_cbmc_info(&self) -> Result<CbmcInfo> {
        // Shares the single `cbmc --version` probe with the version-pin check.
        let version =
            crate::version::cbmc_version_on_path().unwrap_or_else(|| "unknown".to_string());

        // For OS info, we'll use the system information since CBMC --version doesn't provide it
        let os_info = format!(
            "{} {} {}",
            std::env::consts::ARCH,
            std::env::consts::OS,
            std::env::consts::FAMILY
        );

        Ok(CbmcInfo { version, os_info })
    }
}

/// Collect the statistics CBMC reports for a single verification run.
///
/// CBMC reports these as free-text status messages. `--json-ui`, which Kani always passes, wraps
/// each message in a JSON envelope carrying `messageType` and `messageText`, but it does not break
/// the numbers out into fields of their own, so the message text remains the only source available
/// (CBMC's `structured_datat` mechanism, which would render as real JSON fields, is not used by
/// these call sites). What the envelope does buy us is the ability to require a status message and
/// to anchor on CBMC's exact label, rather than searching arbitrary output for a loose pattern.
///
/// Returns `None` when no message carried statistics, which is the case whenever CBMC did not get
/// far enough to report any.
fn merge_cbmc_stats(items: &[ParserItem]) -> Option<CbmcStats> {
    let mut stats = CbmcStats::default();
    let mut found_any = false;

    for item in items {
        if let ParserItem::Message { message_text, message_type } = item
            && message_type == "STATUS-MESSAGE"
        {
            found_any |= record_cbmc_stat(message_text, &mut stats);
        }
    }

    found_any.then_some(stats)
}

/// Record the statistic a single CBMC status message carries, if it carries one. Later messages win,
/// matching CBMC's own behaviour of reporting a running figure more than once.
/// Returns whether this message was recognized.
fn record_cbmc_stat(message: &str, stats: &mut CbmcStats) -> bool {
    // "Generated 1 VCC(s), 1 remaining after simplification"
    if let Some(counts) = message
        .strip_prefix("Generated ")
        .and_then(|rest| rest.strip_suffix(" remaining after simplification"))
        && let Some((generated, remaining)) = counts.split_once(" VCC(s), ")
    {
        stats.vccs_generated = generated.parse().ok();
        stats.vccs_remaining = remaining.parse().ok();
        return stats.vccs_generated.is_some() || stats.vccs_remaining.is_some();
    }

    // "slicing removed 81 assignments", or "simple slicing removed 5 assignments" when only the
    // simple slicer ran. CBMC emits one or the other; our schema has a single field for both.
    if let Some(rest) = message.strip_suffix(" assignments")
        && let Some(count) = rest
            .strip_prefix("slicing removed ")
            .or_else(|| rest.strip_prefix("simple slicing removed "))
    {
        stats.slicing_removed_assignments = count.parse().ok();
        return stats.slicing_removed_assignments.is_some();
    }

    // Everything else is reported as "<label>: <value>".
    let Some((label, value)) = message.split_once(": ") else {
        return false;
    };
    match label {
        // "150 steps"
        "size of program expression" => {
            stats.size_program_expression =
                value.strip_suffix(" steps").and_then(|steps| steps.parse().ok());
            stats.size_program_expression.is_some()
        }
        "Runtime Symex" => record_seconds(value, &mut stats.runtime_symex_s),
        "Runtime Postprocess Equation" => {
            record_seconds(value, &mut stats.runtime_postprocess_equation_s)
        }
        "Runtime Convert SSA" => record_seconds(value, &mut stats.runtime_convert_ssa_s),
        "Runtime Post-process" => record_seconds(value, &mut stats.runtime_post_process_s),
        "Runtime Solver" => record_seconds(value, &mut stats.runtime_solver_s),
        "Runtime decision procedure" => {
            record_seconds(value, &mut stats.runtime_decision_procedure_s)
        }
        _ => false,
    }
}

/// Record a duration CBMC reports as "0.00408627s" or "1.5416e-05s".
fn record_seconds(value: &str, field: &mut Option<f64>) -> bool {
    *field = value.strip_suffix('s').and_then(|seconds| seconds.parse().ok());
    field.is_some()
}

/// The prefix of the report line of a run killed at its policy deadline
/// ([`ExitStatus::Deadline`]), followed by the deadline in milliseconds.
pub const DEADLINE_LINE: &str = "CBMC exceeded its deadline of ";

/// We will use Cadical by default since it performed better than MiniSAT in our analysis.
/// Note: Kissat was marginally better, but it is an external solver which could be more unstable.
static DEFAULT_SOLVER: CbmcSolver = CbmcSolver::Cadical;

#[derive(Clone, Copy, Debug, Display, PartialEq, Eq)]
pub enum VerificationStatus {
    Success,
    Failure,
}

/// Represents failed properties in three different categories.
/// This simplifies the process to determine and format verification results.
#[derive(Clone, Copy, Debug)]
pub enum FailedProperties {
    // No failures
    None,
    // One or more panic-related failures
    PanicsOnly,
    // One or more failures that aren't panic-related
    Other,
    // One or more properties resulted in an ERROR rather than a failing/successful verification
    Error,
}

/// How a CBMC run that produced no results ended. An exit code and a signal are kept apart, and
/// neither is read as running out of memory: nothing CBMC or the operating system reports here
/// establishes what ended the process.
#[derive(Clone, Debug)]
pub enum ExitStatus {
    /// The run reached `--harness-timeout` and was killed.
    Timeout,
    /// The run reached the deadline its `--loop-bounds` policy gives the harness and was killed;
    /// the last status message CBMC reported before it, `None` when it reported none.
    Deadline { milliseconds: u64, last_status_message: Option<String> },
    /// CBMC exited with this code.
    Exited(i32),
    /// CBMC was terminated by this signal.
    Signaled(i32),
}

/// Our (kani-driver) notions of CBMC results.
#[derive(Debug)]
pub struct VerificationResult {
    /// Whether verification should be considered to have succeeded, or have failed.
    pub status: VerificationStatus,
    /// The compact representation for failed properties
    pub failed_properties: FailedProperties,
    /// The `Result` properties in detail or the exit_status of CBMC.
    /// Note: CBMC process exit status is only potentially useful if `status` is `Failure`.
    /// Kani will see CBMC report "failure" that's actually success (interpreting "failed"
    /// checks like coverage as expected and desirable.)
    pub results: Result<Vec<Property>, ExitStatus>,
    /// The runtime duration of this CBMC invocation.
    pub runtime: Duration,
    /// Whether concrete playback generated a test
    pub generated_concrete_test: bool,
    /// The number of quantifier expressions CBMC's solver backend could not encode and
    /// dropped (replaced with unconstrained values), c.f. CBMC's "warning: ignoring forall"
    /// messages. A nonzero count makes the analysis unsound -- an `assume` containing such a
    /// quantifier is not enforced (a successful result would be vacuous), and an `assert`
    /// containing one may fail spuriously -- so it forces the verification to fail (see
    /// `VerificationResult::from`).
    pub ignored_quantifiers: usize,
    /// The coverage results
    pub coverage_results: Option<CoverageResults>,
    /// CBMC execution statistics extracted from messages
    pub cbmc_stats: Option<CbmcStats>,
}

impl KaniSession {
    /// Verify a goto binary that's been prepared with goto-instrument
    pub fn run_cbmc(&self, file: &Path, harness: &HarnessMetadata) -> Result<VerificationResult> {
        // Under `--loop-bounds`, the harness's evidence is reserved and `file`, as instrumentation
        // left it, preserved into it before anything reads it; the preserved copy is the one
        // binary the arguments, the loop discovery, the resolution and the solver then name.
        let evidence = match &self.loop_bounds {
            Some(bounds) => {
                let evidence = bounds.reserve(harness)?;
                let artifact = evidence.preserve_artifact(file)?;
                Some((bounds, evidence, artifact))
            }
            None => None,
        };
        let file = evidence.as_ref().map_or(file, |(_, _, artifact)| artifact.path.as_path());
        let mut args: Vec<OsString> = self.cbmc_flags(file, harness)?;

        // The harness's loop bounds, resolved against the preserved binary, each step recorded in
        // the harness's own evidence before the next reads it, and applied to this run's
        // arguments alone (harnesses may be checked in parallel).
        if let Some((bounds, evidence, artifact)) = &evidence
            && let Some((lines, unwindset)) = bounds.apply(harness, artifact, evidence)?
        {
            for line in lines {
                self.emit_harness_line(&line);
            }
            args.extend(unwindset);
        }

        // The solver, resolved once on the PATH the command would search, the bundle's own
        // first, and canonicalized: the command spawns this executable, so its record names
        // what ran.
        let found = which::which("cbmc")
            .map_err(|error| anyhow::anyhow!("Failed to find cbmc: {error}"))?;
        let solver = found
            .canonicalize()
            .map_err(|error| anyhow::anyhow!("Failed to resolve {}: {error}", found.display()))?;
        let mut cmd = TokioCommand::new(&solver);
        cmd.args(args);

        // `--loop-bounds` refuses `old` in argument validation, so a run with evidence always takes
        // the piped executor, which records the launch at its spawn.
        let verification_results = if self.args.output_format() == crate::args::OutputFormat::Old {
            if self.run_terminal_timeout(cmd).is_err() {
                VerificationResult::mock_failure()
            } else {
                VerificationResult::mock_success()
            }
        } else {
            // Add extra argument to receive the output in JSON format.
            // Done here because now removed `--visualize` used the XML format instead.
            // TODO: move this now that we don't use --visualize
            cmd.arg("--json-ui");

            let evidence = evidence.as_ref().map(|(_, evidence, _)| evidence);
            self.runtime.block_on(self.run_cbmc_piped(cmd, harness, file, evidence))?
        };

        Ok(verification_results)
    }

    async fn run_cbmc_piped(
        &self,
        mut cmd: TokioCommand,
        harness: &HarnessMetadata,
        file: &Path,
        evidence: Option<&crate::loop_bounds::HarnessEvidence>,
    ) -> Result<VerificationResult> {
        if self.args.common_args.verbose() {
            println!("[Kani] Running: `{}`", render_command(cmd.as_std()).to_string_lossy());
        }
        cmd.stdout(std::process::Stdio::piped());
        // The command is complete: the launch attempt is recorded from it, immediately before it
        // is spawned, and a record that cannot be written stops the run before the spawn.
        if let Some(evidence) = evidence {
            let command = cmd.as_std();
            let arguments: Vec<OsString> = command.get_args().map(OsStr::to_os_string).collect();
            evidence.launch(command.get_program(), &arguments)?;
            self.emit_harness_line(&format!(
                "{}{} {}",
                crate::loop_bounds::LAUNCH_LINE,
                harness.pretty_name,
                file.display()
            ));
        }
        // Under `--loop-bounds`, the harness's own deadline from the policy, which requires one of
        // every harness verified and is refused beside `--harness-timeout`; otherwise that option.
        let deadline = self
            .loop_bounds
            .as_ref()
            .map(|bounds| {
                bounds.policy.deadline(&harness.pretty_name).cloned().ok_or_else(|| {
                    anyhow::anyhow!("harness `{}` has no deadline", harness.pretty_name)
                })
            })
            .transpose()?;
        let limit: Option<Duration> = match &deadline {
            Some(deadline) => Some(Duration::from_millis(deadline.milliseconds)),
            None => self.args.harness_timeout.map(Into::into),
        };
        let filter = |item: ParserItem| {
            kani_cbmc_output_filter(
                item,
                self.args.extra_pointer_checks,
                self.args.common_args.quiet,
                &self.args.output_format(),
                self.args.log_file.as_ref(),
            )
        };

        let supervised = supervise(cmd, limit, filter).await?;
        if let (Some(evidence), Some(deadline)) = (evidence, &deadline) {
            evidence.execution(
                deadline.milliseconds,
                supervised.elapsed,
                &supervised.termination,
            )?;
        }
        verification_of(
            supervised,
            deadline.map(|deadline| deadline.milliseconds),
            harness.attributes.should_panic,
            self.args.export_json.is_some(),
        )
    }

    /// "Internal," but also used by call_cbmc_viewer
    pub fn cbmc_flags(
        &self,
        file: &Path,
        harness_metadata: &HarnessMetadata,
    ) -> Result<Vec<OsString>> {
        let mut args = self.cbmc_check_flags();

        if let Some(object_bits) = self.args.cbmc_object_bits() {
            args.push("--object-bits".into());
            args.push(object_bits.to_string().into());
        }

        if let Some(unwind_value) = resolve_unwind_value(&self.args, harness_metadata) {
            args.push("--unwind".into());
            args.push(unwind_value.to_string().into());
        }

        self.handle_solver_args(&harness_metadata.attributes.solver, &mut args)?;

        if self.args.run_sanity_checks {
            args.push("--validate-goto-model".into());
            args.push("--validate-ssa-equation".into());
        }

        if self.args.concrete_playback.is_none() && !self.args.no_slice_formula {
            args.push("--slice-formula".into());
        }

        if self.args.concrete_playback.is_some() {
            args.push("--trace".into());
            // Concrete playback only consumes the values of `kani::any_raw_*`
            // return-value assignments from the trace. CBMC's compact trace
            // retains those (they are regular, non-hidden assignments) while
            // dropping hidden instrumentation steps whose values can dominate
            // the trace by orders of magnitude on contract-heavy harnesses
            // (e.g. 427 MB -> 3 MB of JSON). Requires CBMC with
            // https://github.com/diffblue/cbmc/pull/9135 to have an effect;
            // CBMC versions that do not yet honor `--compact-trace` with
            // `--json-ui` accept but ignore the option, so this is
            // compatible either way.
            args.push("--compact-trace".into());
        }

        args.extend(self.args.cbmc_args.iter().cloned());

        args.push(file.to_owned().into_os_string());

        // Make CBMC verbose by default to tell users about unwinding progress. This should be
        // reviewed as CBMC's verbosity defaults evolve.
        args.push("--verbosity".into());
        args.push("9".into());

        Ok(args)
    }

    /// Just the flags to CBMC that enable property checking of any sort.
    pub fn cbmc_check_flags(&self) -> Vec<OsString> {
        let mut args = Vec::new();

        // We assume that malloc cannot fail, see https://github.com/model-checking/kani/issues/891
        args.push("--no-malloc-may-fail".into());

        // With PR #2630 we generate the appropriate checks directly rather than relying on CBMC's
        // checks (which are for C semantics).
        args.push("--no-undefined-shift-check".into());
        // With PR #647 we use Rust's `-C overflow-checks=on` instead of:
        // --unsigned-overflow-check
        // --signed-overflow-check
        // So these options are deliberately skipped to avoid erroneously re-checking operations.
        args.push("--no-signed-overflow-check".into());

        if !self.args.checks.memory_safety_on() {
            args.push("--no-bounds-check".into());
            args.push("--no-pointer-check".into());
        }
        if self.args.checks.overflow_on() {
            args.push("--nan-check".into());

            // TODO: Implement conversion checks as an optional check.
            // They are a well defined operation in rust, but they may yield unexpected results to
            // many users. https://github.com/model-checking/kani/issues/840
            // We might want to create a transformation pass instead of enabling CBMC since Kani
            // compiler sometimes rely on the bitwise conversion of signed <-> unsigned.
            // args.push("--conversion-check".into());
        } else {
            args.push("--no-div-by-zero-check".into());
        }

        if !self.args.checks.unwinding_on() {
            args.push("--no-unwinding-assertions".into());
        } else {
            args.push("--no-self-loops-to-assumptions".into());
        }

        if self.args.extra_pointer_checks {
            // This was adding a lot of false positives with std dangling pointer. We should
            // still catch any invalid dereference with --pointer-check. Thus, only enable them
            // if the user explicitly request them.
            args.push("--pointer-overflow-check".into());
        } else {
            args.push("--no-pointer-primitive-check".into());
        }

        args
    }

    /// The solver this run will actually use for a harness: `--solver` takes precedence over the
    /// harness attribute, which takes precedence over the default.
    ///
    /// Anything reporting the configuration of a run must resolve it through here rather than
    /// reading the harness attribute directly, or it will describe a different run than the one
    /// `handle_solver_args` builds.
    pub fn resolved_solver<'a>(&'a self, harness_solver: &'a Option<CbmcSolver>) -> &'a CbmcSolver {
        self.args.solver.as_ref().or(harness_solver.as_ref()).unwrap_or(&DEFAULT_SOLVER)
    }

    pub fn handle_solver_args(
        &self,
        harness_solver: &Option<CbmcSolver>,
        args: &mut Vec<OsString>,
    ) -> Result<()> {
        let solver = self.resolved_solver(harness_solver);

        match solver {
            CbmcSolver::Bitwuzla => {
                args.push("--bitwuzla".into());
            }
            CbmcSolver::Cadical => {
                args.push("--sat-solver".into());
                args.push("cadical".into());
            }
            CbmcSolver::Cvc5 => {
                args.push("--cvc5".into());
            }
            CbmcSolver::Kissat => {
                args.push("--external-sat-solver".into());
                args.push("kissat".into());
            }
            CbmcSolver::Minisat => {
                // Minisat is currently CBMC's default solver, so no need to
                // pass any arguments
            }
            CbmcSolver::Z3 => {
                args.push("--z3".into());
            }
            CbmcSolver::Binary(solver_binary) => {
                // Check if the specified binary exists in path
                if which::which(solver_binary).is_err() {
                    bail!("the specified solver \"{solver_binary}\" was not found in path")
                }
                args.push("--external-sat-solver".into());
                args.push(solver_binary.into());
            }
        }
        Ok(())
    }
}

/// One supervised solver execution: CBMC's output when it ran to its end, the time from
/// immediately after its spawn to its end, and how it ended ([`SolverTermination`]).
pub(crate) struct Supervised {
    pub output: Option<VerificationOutput>,
    pub elapsed: Duration,
    pub termination: SolverTermination,
}

/// Spawn `cmd` and read its output through `filter` until it ends or `limit` passes. One
/// monotonic start is taken immediately after the spawn returns, and the absolute deadline is
/// computed from it; the elapsed time ends at normal completion, at the deadline's detection, or at
/// an execution error. At the deadline, CBMC's last status message is kept, the process is killed
/// and reaped, and the time and any error of that cleanup are recorded apart. A spawn that fails
/// is an error: no execution began.
pub(crate) async fn supervise(
    mut cmd: TokioCommand,
    limit: Option<Duration>,
    mut filter: impl FnMut(ParserItem) -> Option<ParserItem>,
) -> Result<Supervised> {
    let mut child = cmd.spawn().map_err(|error| anyhow::anyhow!("Failed to run cbmc: {error}"))?;
    let start = tokio::time::Instant::now();
    let deadline = match limit.map(|limit| start.checked_add(limit)) {
        Some(Some(at)) => Some(at),
        Some(None) => {
            let elapsed = start.elapsed();
            let cleanup = child.kill().await;
            return Ok(Supervised {
                output: None,
                elapsed,
                termination: SolverTermination::ExecutionError {
                    operation: "computing the deadline".to_string(),
                    diagnostic: format!(
                        "the deadline of {limit:?} past the start does not fit the clock{}",
                        cleanup
                            .err()
                            .map(|error| format!("; terminating CBMC: {error}"))
                            .unwrap_or_default()
                    ),
                },
            });
        }
        None => None,
    };
    let last_status = std::cell::RefCell::new(None::<String>);
    let tracked = |item: ParserItem| {
        if let ParserItem::Message { message_text, message_type } = &item
            && message_type == "STATUS-MESSAGE"
        {
            *last_status.borrow_mut() = Some(message_text.clone());
        }
        filter(item)
    };
    let read = match deadline {
        Some(at) => tokio::time::timeout_at(at, process_cbmc_output(&mut child, tracked)).await,
        None => Ok(process_cbmc_output(&mut child, tracked).await),
    };
    let elapsed = start.elapsed();
    Ok(match read {
        Ok(Ok(output)) => Supervised {
            termination: match output.termination {
                ProcessTermination::Exited { code } => SolverTermination::Exited { code },
                ProcessTermination::Signaled { signal } => SolverTermination::Signaled { signal },
            },
            output: Some(output),
            elapsed,
        },
        Ok(Err(error)) => Supervised {
            output: None,
            elapsed,
            termination: SolverTermination::ExecutionError {
                operation: "reading CBMC's output and reaping it".to_string(),
                diagnostic: format!("{error:#}"),
            },
        },
        Err(_) => {
            let last_status_message = last_status.into_inner();
            let detected = tokio::time::Instant::now();
            let cleanup = child.kill().await;
            Supervised {
                output: None,
                elapsed,
                termination: SolverTermination::Deadline {
                    last_status_message,
                    cleanup_elapsed: detected.elapsed().into(),
                    cleanup_error: cleanup.err().map(|error| error.to_string()),
                },
            }
        }
    })
}

/// The verification result of a supervised execution. Only a normal exit permits its properties
/// to be interpreted, whatever its exit code (CBMC exits nonzero for a harness Kani verifies, its
/// reachability checks reported as failures). A run terminated by a signal has no verdict
/// whatever it printed: its properties may be incomplete or the signal may have interrupted a
/// consistent run, and neither can qualify a proof or detect a planted defect; they stay in the
/// run's log as diagnostics. A run killed at its `deadline` (`--loop-bounds`) or
/// `--harness-timeout` has no verdict. A deadline whose cleanup failed, and an execution error,
/// are errors: no result is accepted from them.
pub(crate) fn verification_of(
    supervised: Supervised,
    deadline: Option<u64>,
    should_panic: bool,
    collect_cbmc_stats: bool,
) -> Result<VerificationResult> {
    let results = match supervised.termination {
        SolverTermination::Exited { code } => {
            let output = supervised
                .output
                .ok_or_else(|| anyhow::anyhow!("a CBMC run that exited has no output"))?;
            return Ok(VerificationResult::from(
                output.processed_items,
                code,
                should_panic,
                supervised.elapsed,
                collect_cbmc_stats,
            ));
        }
        SolverTermination::Signaled { signal } => Err(ExitStatus::Signaled(signal)),
        SolverTermination::Deadline { cleanup_error: Some(error), .. } => {
            bail!("CBMC could not be terminated at its deadline: {error}")
        }
        SolverTermination::Deadline { last_status_message, cleanup_error: None, .. } => {
            match deadline {
                Some(milliseconds) => {
                    Err(ExitStatus::Deadline { milliseconds, last_status_message })
                }
                None => Err(ExitStatus::Timeout),
            }
        }
        SolverTermination::ExecutionError { operation, diagnostic } => {
            bail!("CBMC's execution failed {operation}: {diagnostic}")
        }
    };
    Ok(VerificationResult {
        status: VerificationStatus::Failure,
        failed_properties: FailedProperties::None,
        results,
        runtime: supervised.elapsed,
        generated_concrete_test: false,
        ignored_quantifiers: 0,
        coverage_results: None,
        cbmc_stats: None,
    })
}

/// Count CBMC messages reporting that a quantifier expression could not be encoded and was
/// dropped. CBMC's SAT-based backends only support quantifiers with constant bounds; other
/// quantifiers are replaced by unconstrained values, with only a low-visibility message
/// (`prop_conv_solvert::ignoring`, printed as "warning: ignoring forall" followed by the
/// pretty-printed expression).
fn count_ignored_quantifiers(items: &[ParserItem]) -> usize {
    items
        .iter()
        .filter(|item| {
            matches!(item, ParserItem::Message { message_text, .. }
                if message_text.starts_with("warning: ignoring forall")
                    || message_text.starts_with("warning: ignoring exists"))
        })
        .count()
}

/// The error rendered when the solver backend dropped quantifier expressions.
fn ignored_quantifiers_error(count: usize) -> String {
    format!(
        "error: the solver backend does not support quantifiers with non-constant bounds \
and ignored {count} quantifier expression(s), replacing them with unconstrained values.\n\
         Kani cannot soundly verify this harness: `kani::assume` calls containing such a \
quantifier are NOT enforced (a successful result would not cover the intended property), and \
`kani::assert` calls containing one may fail spuriously.\n\
         Use an SMT solver backend that supports quantifiers, e.g. `#[kani::solver(z3)]`.\n"
    )
}

impl VerificationResult {
    /// Computes a `VerificationResult` (kani-driver's notion of the result of a CBMC call) from
    /// the items of a CBMC run that exited by itself with `exit_code` (cbmc_output_parser's idea
    /// of CBMC results). Only a normal exit reaches here: a run terminated by a signal gives no
    /// verdict whatever it printed ([`verification_of`]).
    ///
    /// NOTE: We actually ignore the CBMC exit code, in favor of two checks:
    ///   1. Examining the actual results of CBMC properties.
    ///      (CBMC will regularly report "failure" but that's just our cover checks.)
    ///   2. Positively checking for the presence of results.
    ///      (Do not mistake lack of results for success: report it as failure.)
    fn from(
        processed_items: Vec<ParserItem>,
        exit_code: i32,
        should_panic: bool,
        runtime: Duration,
        collect_cbmc_stats: bool,
    ) -> VerificationResult {
        let ignored_quantifiers = count_ignored_quantifiers(&processed_items);
        let (remaining_items, results) = extract_results(processed_items);

        // Only `--export-json` consumes these, and collecting them means running several regexes
        // over every message CBMC emitted, so skip the work entirely when nothing will read it.
        let cbmc_stats = if collect_cbmc_stats { merge_cbmc_stats(&remaining_items) } else { None };

        if let Some(results) = results {
            let (mut status, mut failed_properties) =
                verification_outcome_from_properties(&results, should_panic);
            // A dropped quantifier makes the analysis unsound: a `kani::assume` containing one is
            // silently not enforced, so a "successful" result may be vacuous. Kani must never
            // report success in that case -- force a failure and (via the rendered error) direct
            // the user to an SMT backend that supports quantifiers.
            if ignored_quantifiers > 0 {
                status = VerificationStatus::Failure;
                failed_properties = FailedProperties::Error;
            }
            let coverage_results = coverage_results_from_properties(&results);
            VerificationResult {
                status,
                failed_properties,
                results: Ok(results),
                runtime,
                generated_concrete_test: false,
                ignored_quantifiers,
                coverage_results,
                cbmc_stats,
            }
        } else {
            // We never got results from CBMC - something went wrong (e.g. crash) so it's failure
            VerificationResult {
                status: VerificationStatus::Failure,
                failed_properties: FailedProperties::Other,
                results: Err(ExitStatus::Exited(exit_code)),
                runtime,
                generated_concrete_test: false,
                ignored_quantifiers,
                coverage_results: None,
                cbmc_stats,
            }
        }
    }

    pub fn mock_success() -> VerificationResult {
        VerificationResult {
            status: VerificationStatus::Success,
            failed_properties: FailedProperties::None,
            results: Ok(vec![]),
            runtime: Duration::from_secs(0),
            generated_concrete_test: false,
            ignored_quantifiers: 0,
            coverage_results: None,
            cbmc_stats: None,
        }
    }

    fn mock_failure() -> VerificationResult {
        VerificationResult {
            status: VerificationStatus::Failure,
            failed_properties: FailedProperties::Other,
            // on failure, exit codes in theory might be used,
            // but `mock_failure` should never be used in a context where they will,
            // so again use something weird:
            results: Err(ExitStatus::Exited(42)),
            runtime: Duration::from_secs(0),
            generated_concrete_test: false,
            ignored_quantifiers: 0,
            coverage_results: None,
            cbmc_stats: None,
        }
    }

    pub fn render(&self, output_format: &OutputFormat, should_panic: bool) -> String {
        match &self.results {
            Ok(results) => {
                let status = self.status;
                let failed_properties = self.failed_properties;
                let show_checks = matches!(output_format, OutputFormat::Regular);

                let mut result = if let Some(cov_results) = &self.coverage_results {
                    format_coverage(
                        results,
                        cov_results,
                        status,
                        should_panic,
                        failed_properties,
                        show_checks,
                    )
                } else {
                    format_result(results, status, should_panic, failed_properties, show_checks)
                };
                if self.ignored_quantifiers > 0 {
                    let error = ignored_quantifiers_error(self.ignored_quantifiers);
                    // Surface the soundness error immediately before the overall
                    // `VERIFICATION:- ...` line, so it explains the forced failure. Fall back to
                    // appending it (e.g. coverage output, which has no such line) if the marker
                    // isn't present.
                    match result.find("\nVERIFICATION:- ") {
                        Some(pos) => result.insert_str(pos + 1, &error),
                        None => {
                            result.push('\n');
                            result.push_str(&error);
                        }
                    }
                }
                writeln!(result, "Verification Time: {}s", self.runtime.as_secs_f32()).unwrap();
                result
            }
            Err(exit_status) => {
                let verification_result = console::style("FAILED").red();
                let (header, explanation) = match exit_status {
                    ExitStatus::Timeout => (
                        String::from("CBMC failed"),
                        String::from(
                            "CBMC reached --harness-timeout and the harness has no verdict. The \
                             run cost more than the limit: right-size the harness's domain or the \
                             code it reasons about, or find the defect that made it cost more.\n",
                        ),
                    ),
                    ExitStatus::Deadline { milliseconds, last_status_message } => (
                        format!("{DEADLINE_LINE}{milliseconds} ms"),
                        format!(
                            "The run was killed at its deadline and the harness has no verdict. \
                             It cost more than its policy allows over its supported expectation: \
                             the harness is wrongly sized for the expectation, the expectation is \
                             wrong, or the code it checks regressed; an expectation is never \
                             raised solely to turn the run green. CBMC's last status message: \
                             {}\n",
                            last_status_message.as_deref().unwrap_or("none")
                        ),
                    ),
                    ExitStatus::Exited(code) => {
                        (format!("CBMC failed with status {code}"), String::new())
                    }
                    ExitStatus::Signaled(signal) => (
                        format!("CBMC failed: terminated by signal {signal}"),
                        String::from(
                            "The signal's sender is not established by its number; the harness \
                             has no verdict.\n",
                        ),
                    ),
                };
                format!(
                    "\n{header}\n\
                    VERIFICATION:- {verification_result}\n\
                    {explanation}",
                )
            }
        }
    }
}

/// We decide if verification succeeded based on properties, not (typically) on exit code
fn verification_outcome_from_properties(
    properties: &[Property],
    should_panic: bool,
) -> (VerificationStatus, FailedProperties) {
    let failed_properties = determine_failed_properties(properties);
    let status = if should_panic {
        match failed_properties {
            FailedProperties::Error => VerificationStatus::Failure,
            FailedProperties::None | FailedProperties::Other => VerificationStatus::Failure,
            FailedProperties::PanicsOnly => VerificationStatus::Success,
        }
    } else {
        match failed_properties {
            FailedProperties::Error => VerificationStatus::Failure,
            FailedProperties::None => VerificationStatus::Success,
            FailedProperties::PanicsOnly | FailedProperties::Other => VerificationStatus::Failure,
        }
    };
    (status, failed_properties)
}

/// Determines the `FailedProperties` variant that corresponds to an array of properties
fn determine_failed_properties(properties: &[Property]) -> FailedProperties {
    if properties.iter().any(|prop| prop.status == CheckStatus::Error) {
        return FailedProperties::Error;
    };
    let failed_properties: Vec<&Property> =
        properties.iter().filter(|prop| prop.status == CheckStatus::Failure).collect();
    // Return `FAILURE` if there isn't at least one failed property
    if failed_properties.is_empty() {
        FailedProperties::None
    } else {
        // Check if all failed properties correspond to the `assertion` class.
        // Note: Panics caused by `panic!` and `assert!` fall into this class.
        let all_failed_checks_are_panics =
            failed_properties.iter().all(|prop| prop.property_class() == "assertion");
        if all_failed_checks_are_panics {
            FailedProperties::PanicsOnly
        } else {
            FailedProperties::Other
        }
    }
}

fn coverage_results_from_properties(properties: &[Property]) -> Option<CoverageResults> {
    let cov_properties: Vec<&Property> =
        properties.iter().filter(|p| p.is_code_coverage_property()).collect();

    if cov_properties.is_empty() {
        return None;
    }

    // Postprocessing the coverage results involves matching on the descriptions
    // of code coverage properties with the `counter_re` regex. These are two
    // real examples of such descriptions:
    //
    // ```
    // CounterIncrement(0) $test_cov$ - src/main.rs:5:1 - 6:15
    // ExpressionUsed(0) $test_cov$ - src/main.rs:6:19 - 6:28
    // ```
    //
    // The span is further processed to extract the code region attributes.
    // Ideally, we should have coverage mappings (i.e., the relation between
    // counters and code regions) available in the coverage metadata:
    // <https://github.com/model-checking/kani/issues/3445>. If that were the
    // case, we would not need the spans in these descriptions.
    let counter_re = {
        static COUNTER_RE: OnceLock<Regex> = OnceLock::new();
        COUNTER_RE.get_or_init(|| {
            Regex::new(
                r#"^(?<kind>VirtualCounter\(bcb)(?<counter_num>[0-9]+)\) \$(?<func_name>[^\$]+)\$ - (?<span>.+)"#,
            )
            .unwrap()
        })
    };

    let mut coverage_results: BTreeMap<String, Vec<CoverageCheck>> = BTreeMap::default();

    for prop in cov_properties {
        let mut prop_processed = false;
        if let Some(captures) = counter_re.captures(&prop.description) {
            let counter_num = &captures["counter_num"];
            let function = demangle(&captures["func_name"]).to_string();
            let status = prop.status;
            let span = captures["span"].to_string();

            let counter_id = counter_num.parse().unwrap();
            let term = CoverageTerm::Counter(counter_id);
            let region = CoverageRegion::from_str(span);

            let cov_check = CoverageCheck::new(function, term, region, status);
            let file = cov_check.region.file.clone();

            if let Entry::Vacant(e) = coverage_results.entry(file.clone()) {
                e.insert(vec![cov_check]);
            } else {
                coverage_results.entry(file).and_modify(|checks| checks.push(cov_check));
            }
            prop_processed = true;
        }

        assert!(prop_processed, "error: coverage property not processed\n{prop:?}");
    }

    Some(CoverageResults::new(coverage_results))
}
/// Solve Unwind Value from conflicting inputs of unwind values. (--default-unwind, annotation-unwind, --unwind)
pub fn resolve_unwind_value(
    args: &VerificationArgs,
    harness_metadata: &HarnessMetadata,
) -> Option<u32> {
    // Check for which flag is being passed and prioritize extracting unwind from the
    // respective flag/annotation.
    args.unwind.or(harness_metadata.attributes.unwind_value).or(args.default_unwind)
}

#[cfg(test)]
mod tests {
    use crate::args;
    use crate::metadata::tests::mock_proof_harness;
    use clap::Parser;

    use super::*;

    /// The statistics messages below are verbatim CBMC 6.x `--json-ui` output, so a CBMC change to
    /// any of these labels shows up here as a test failure rather than as silently missing data.
    #[test]
    fn check_cbmc_stats_from_status_messages() {
        let messages = [
            "Runtime Symex: 0.00049675s",
            "size of program expression: 21 steps",
            "simple slicing removed 5 assignments",
            "Generated 1 VCC(s), 1 remaining after simplification",
            "Runtime Postprocess Equation: 1.5416e-05s",
            "Runtime Convert SSA: 0.00012525s",
            "Runtime Post-process: 2.0292e-05s",
            "Runtime Solver: 4.1167e-05s",
            "Runtime decision procedure: 0.000193542s",
        ];
        let items: Vec<ParserItem> = messages
            .iter()
            .map(|text| ParserItem::Message {
                message_text: text.to_string(),
                message_type: "STATUS-MESSAGE".to_string(),
            })
            .collect();

        let stats = merge_cbmc_stats(&items).expect("statistics should be recognized");
        assert_eq!(stats.runtime_symex_s, Some(0.00049675));
        assert_eq!(stats.size_program_expression, Some(21));
        assert_eq!(stats.slicing_removed_assignments, Some(5));
        assert_eq!(stats.vccs_generated, Some(1));
        assert_eq!(stats.vccs_remaining, Some(1));
        assert_eq!(stats.runtime_postprocess_equation_s, Some(1.5416e-05));
        assert_eq!(stats.runtime_convert_ssa_s, Some(0.00012525));
        assert_eq!(stats.runtime_post_process_s, Some(2.0292e-05));
        assert_eq!(stats.runtime_solver_s, Some(4.1167e-05));
        assert_eq!(stats.runtime_decision_procedure_s, Some(0.000193542));
    }

    /// The full slicer reports without the "simple" prefix.
    #[test]
    fn check_cbmc_stats_full_slicer() {
        let mut stats = CbmcStats::default();
        assert!(record_cbmc_stat("slicing removed 81 assignments", &mut stats));
        assert_eq!(stats.slicing_removed_assignments, Some(81));
    }

    /// Anchoring on the label is what the message type and exact-match parsing buy us: text that
    /// merely mentions a statistic, or that CBMC reports as a warning rather than a status message,
    /// must not be mistaken for a measurement.
    #[test]
    fn check_cbmc_stats_ignore_unrelated_text() {
        let mut stats = CbmcStats::default();
        assert!(!record_cbmc_stat("assertion failed: Runtime Solver: 1s is too slow", &mut stats));
        assert!(!record_cbmc_stat("Runtime Solver: not-a-number", &mut stats));
        assert!(!record_cbmc_stat("VERIFICATION FAILED", &mut stats));
        assert_eq!(stats.runtime_solver_s, None);

        let warning = [ParserItem::Message {
            message_text: "Runtime Solver: 4.1167e-05s".to_string(),
            message_type: "WARNING".to_string(),
        }];
        assert!(merge_cbmc_stats(&warning).is_none());
    }

    /// No statistics at all (CBMC died early, or verbosity hid them) must not fabricate a record.
    #[test]
    fn check_cbmc_stats_absent() {
        assert!(merge_cbmc_stats(&[]).is_none());
    }

    /// `script` run by `sh` under `limit`, supervised as CBMC is.
    fn supervised(script: &str, limit: Option<Duration>) -> Supervised {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let mut cmd = TokioCommand::new("sh");
        cmd.args(["-c", script]).stdout(std::process::Stdio::piped());
        runtime.block_on(supervise(cmd, limit, Some)).unwrap()
    }

    /// An exit code of 137 and a SIGKILL are different terminations, each kept as it happened,
    /// and neither is reported as running out of memory.
    #[test]
    fn an_exit_and_a_signal_are_kept_apart() {
        let exited = supervised("exit 137", None);
        assert_eq!(exited.termination, SolverTermination::Exited { code: 137 });
        let result = verification_of(exited, None, false, false).unwrap();
        assert!(matches!(result.results, Err(ExitStatus::Exited(137))));
        let rendered = result.render(&OutputFormat::Regular, false);
        assert!(rendered.contains("CBMC failed with status 137"), "{rendered}");
        assert!(!rendered.contains("memory"), "{rendered}");
        let killed = supervised("kill -9 $$", None);
        assert_eq!(killed.termination, SolverTermination::Signaled { signal: 9 });
        let result = verification_of(killed, None, false, false).unwrap();
        assert!(matches!(result.results, Err(ExitStatus::Signaled(9))));
        let rendered = result.render(&OutputFormat::Regular, false);
        assert!(rendered.contains("CBMC failed: terminated by signal 9"), "{rendered}");
        assert!(!rendered.contains("memory"), "{rendered}");
    }

    /// A run that outlives its limit is detected at it, killed and reaped, the time to its
    /// detection and the time of its cleanup measured apart; under a policy deadline it is a
    /// deadline, otherwise `--harness-timeout`'s timeout, and either has no verdict.
    #[test]
    fn a_deadline_is_detected_then_cleaned_up_apart() {
        let limit = Duration::from_millis(100);
        let found = supervised("sleep 30", Some(limit));
        assert!(
            found.elapsed >= limit && found.elapsed < Duration::from_secs(30),
            "{:?}",
            found.elapsed
        );
        let SolverTermination::Deadline { cleanup_error: None, cleanup_elapsed, .. } =
            &found.termination
        else {
            panic!("not a cleaned-up deadline: {:?}", found.termination);
        };
        assert!(Duration::new(cleanup_elapsed.seconds, cleanup_elapsed.nanoseconds) < limit);
        let result = verification_of(found, Some(100), false, false).unwrap();
        assert!(matches!(
            result.results,
            Err(ExitStatus::Deadline { milliseconds: 100, last_status_message: None })
        ));
        assert_eq!(result.status, VerificationStatus::Failure);
        let timed_out = verification_of(supervised("sleep 30", Some(limit)), None, false, false);
        assert!(matches!(timed_out.unwrap().results, Err(ExitStatus::Timeout)));
    }

    /// CBMC's result item of one assertion with `status`, as `--json-ui` reports it.
    fn properties(status: &str) -> Vec<ParserItem> {
        let item = serde_json::from_str(&format!(
            r#"{{"result": [{{"description": "assertion failed: x", "property": "h.assertion.1",
                "sourceLocation": {{"file": "a.rs", "function": "h", "line": "1", "column": "1"}},
                "status": "{status}"}}]}}"#
        ))
        .unwrap();
        vec![item]
    }

    /// A supervised run that ended by `termination`, CBMC having printed `items`.
    fn ended(termination: ProcessTermination, items: Vec<ParserItem>) -> Supervised {
        Supervised {
            termination: match termination {
                ProcessTermination::Exited { code } => SolverTermination::Exited { code },
                ProcessTermination::Signaled { signal } => SolverTermination::Signaled { signal },
            },
            output: Some(VerificationOutput { termination, processed_items: items }),
            elapsed: Duration::from_millis(7),
        }
    }

    /// Only a normal exit permits property interpretation, whatever its code: exit 10 with
    /// successful properties verifies. A signal gives no verdict whatever CBMC printed before
    /// it, successful properties, a failed assertion or none: never a proof and never an
    /// attributable failure.
    #[test]
    fn only_a_normal_exit_interprets_properties() {
        let verified = verification_of(
            ended(ProcessTermination::Exited { code: 10 }, properties("SUCCESS")),
            None,
            false,
            false,
        )
        .unwrap();
        assert_eq!(verified.status, VerificationStatus::Success);
        assert!(verified.results.is_ok());
        for items in [properties("SUCCESS"), properties("FAILURE"), Vec::new()] {
            let found = verification_of(
                ended(ProcessTermination::Signaled { signal: 9 }, items),
                None,
                false,
                false,
            )
            .unwrap();
            assert!(matches!(found.results, Err(ExitStatus::Signaled(9))));
            assert_eq!(found.status, VerificationStatus::Failure);
            assert!(matches!(found.failed_properties, FailedProperties::None));
            assert_eq!(found.runtime, Duration::from_millis(7));
            let rendered = found.render(&OutputFormat::Regular, false);
            assert!(rendered.contains("CBMC failed: terminated by signal 9"), "{rendered}");
            assert!(!rendered.contains("SUCCESSFUL") && !rendered.contains("Failed Checks"));
        }
    }

    /// A deadline whose cleanup failed, and an execution error, are never accepted as a
    /// termination with a result.
    #[test]
    fn a_failed_cleanup_or_execution_is_never_a_result() {
        let cleanup = Supervised {
            output: None,
            elapsed: Duration::from_millis(5),
            termination: SolverTermination::Deadline {
                last_status_message: None,
                cleanup_elapsed: Duration::ZERO.into(),
                cleanup_error: Some("no such process".to_string()),
            },
        };
        let error = verification_of(cleanup, Some(5), false, false).unwrap_err().to_string();
        assert!(error.contains("could not be terminated at its deadline"), "{error}");
        let execution = Supervised {
            output: None,
            elapsed: Duration::ZERO,
            termination: SolverTermination::ExecutionError {
                operation: "reading".to_string(),
                diagnostic: "broken pipe".to_string(),
            },
        };
        let error = verification_of(execution, Some(5), false, false).unwrap_err().to_string();
        assert!(error.contains("CBMC's execution failed reading: broken pipe"), "{error}");
    }

    #[test]
    fn check_resolve_unwind_value() {
        // Command line unwind value for specific harnesses take precedence over default annotation value
        let args_empty = ["kani", "x.rs"];
        let args_only_default = ["kani", "x.rs", "--default-unwind", "2"];
        let args_only_harness = ["kani", "x.rs", "--unwind", "1", "--harness", "check_one"];
        let args_both =
            ["kani", "x.rs", "--default-unwind", "2", "--unwind", "1", "--harness", "check_one"];

        let harness_none = mock_proof_harness("check_one", None, None, None);
        let harness_some = mock_proof_harness("check_one", Some(3), None, None);

        fn resolve(args: &[&str], harness: &HarnessMetadata) -> Option<u32> {
            resolve_unwind_value(
                &args::StandaloneArgs::try_parse_from(args).unwrap().verify_opts,
                harness,
            )
        }

        // test against no unwind annotation
        assert_eq!(resolve(&args_empty, &harness_none), None);
        assert_eq!(resolve(&args_only_default, &harness_none), Some(2));
        assert_eq!(resolve(&args_only_harness, &harness_none), Some(1));
        assert_eq!(resolve(&args_both, &harness_none), Some(1));

        // test against unwind annotation
        assert_eq!(resolve(&args_empty, &harness_some), Some(3));
        assert_eq!(resolve(&args_only_default, &harness_some), Some(3));
        assert_eq!(resolve(&args_only_harness, &harness_some), Some(1));
        assert_eq!(resolve(&args_both, &harness_some), Some(1));
    }
}
