# The fitting-strategy config: optimization, the noise-tempering schedule, and the MCMC
# sampler (AdvancedHMC NUTS, Reactant NUTS, or KineticSamplers MCLMC or WALNUTS).

const KINETIC_ALGORITHMS = ("mclmc", "walnuts")
const KINETIC_WARMUP = Dict("mclmc" => 150_000, "walnuts" => 1_000)
const MCLMC_INTEGRATORS = ("mclachlan", "leapfrog", "baoab")

function _mclmc_integrator_name(name)
    name in MCLMC_INTEGRATORS || error(
        "sampler.integrator must be one of $(join(repr.(MCLMC_INTEGRATORS), ", ")), got $(repr(name))"
    )
    return String(name)
end

function _positive_or_nothing(x, key)
    isnothing(x) && return nothing
    x isa Real && x > 0 || error("$key must be a positive number, got $(repr(x))")
    return Float64(x)
end
const MOVE_KINDS = ("flux_gain", "field_scale", "rho_field", "mean_field", "phase_sheet", "chain_hyper")

"""
    MoveSpec

One `[[sampler.moves]]` table of the fitting config: the move `kind` (one of
`$(MOVE_KINDS)`), the parameters it acts on (`params`, `nothing` for all that apply), the
proposals per call (`rounds`), the warmup acceptance target, and the initial step scale of a
random-walk move (`nothing` for the kind's default). See [`build_moves`](@ref).
"""
Base.@kwdef struct MoveSpec
    kind::String
    params::Union{Nothing, Vector{String}} = nothing
    rounds::Int = 1
    target_accept::Float64 = 0.45
    initial_scale::Union{Nothing, Float64} = nothing
    band_limit::Float64 = Inf
end

"""
    FittingStrategy

Typed description of how to fit: the optimizer, the fractional-noise tempering schedule,
the MCMC sampler and its tuning, and run-level flags (Reactant, benchmark, start).

Note: `restart` is deliberately NOT part of the strategy — resuming a run is a one-off
action passed at call time (the `--restart` CLI flag / `comrade_imager(...; restart=true)`),
not configuration to track in a TOML.
"""
Base.@kwdef struct FittingStrategy
    # optimizer
    opt_method::String = "Adam"     # Adam | AdamW | LBFGS (LBFGS is CPU-only)
    maxiters::Int = 10_000
    ntrials::Int = 5
    g_tol::Float64 = 0.1
    eta::Float64 = 0.001            # learning rate for the Optimisers.jl rules (Adam/AdamW)
    # sky scale parameters held at their prior medians during optimization; non-empty also
    # rescales the optimum's fields to unit rms (see `build_fitting_config`)
    fix_scales::Vector{Symbol} = Symbol[]
    # tempering: fractional-noise level per optimization round (0.0 = full data)
    noise_schedule::Vector{Float64} = [0.05, 0.025, 0.0]
    nsample::Int = 10_000
    nadapt::Int = 5_000
    # NUTS's initial step size; for MCLMC the sampling step size, `nothing` keeping the warmup's.
    step_size::Union{Nothing, Float64} = 0.01
    target_accept::Float64 = 0.9
    init_buffer::Int = 200
    term_buffer::Int = 500
    max_tree_depth::Int = 10
    chunk_size::Int = 100
    # MCLMC: sampling blocks per kept draw
    thin::Int = 1
    # MCLMC: steps per chain of the warmup's final phase, run at the sampling parameters
    final::Int = 0
    # MCLMC and WALNUTS: stiff directions of the inverse mass and the draws each estimate uses
    stiff_directions::Int = 8
    stiff_draws::Int = 2
    # MCLMC: the autocorrelation times the trajectory length follows, "all" recorded rows or
    # "projections" onto the low-rank directions of the inverse mass
    trajectory_rows::String = "all"
    # MCLMC: the quantile of those times the trajectory length follows (`nothing`: their mean),
    # the fraction of the way each refit window moves it, and the number of final windows
    trajectory_quantile::Union{Nothing, Float64} = nothing
    refit_trajectory_step::Float64 = 1.0
    window_iterations::Int = 1
    # "nuts", or "mclmc" or "walnuts" for KineticSamplers chains with `chains` chains and a
    # pooled warmup
    algorithm::String = "nuts"
    chains::Int = 4
    max_step_halvings::Int = 3
    # MCLMC's integrator: "mclachlan", "leapfrog" or "baoab".
    integrator::String = "mclachlan"
    # MCLMC halves a step whose energy error exceeds this, in total; `nothing` keeps
    # KineticSamplers' default of 0.25 √d.
    max_energy_error::Union{Nothing, Float64} = nothing
    # The phase per step at the median curvature of the MCLMC warmup's step-size cap; `nothing`
    # is `MCLACHLAN_CAP_PHASE` for McLachlan (see `mclmc_cap_phase`).
    max_mode_phase::Union{Nothing, Float64} = nothing
    # MCLMC's trajectory length in units of √d; `nothing` sets it from the warmup's window.
    trajectory_length::Union{Nothing, Float64} = nothing
    # MCLMC samples at the step size of `KineticSamplers.tune_step_size` after the warmup.
    bias_tuning::Bool = true
    # Metropolis–Hastings moves run between sampler chunks or blocks (see `build_moves`).
    moves::Vector{MoveSpec} = MoveSpec[]
    # run
    # Latent space optimized and sampled in: "flat" (`asflat`) or "stdnormal" (the
    # ProbabilityTransports `StdNormal` transport, where the prior is N(0, I); needed by
    # priors with no flat transform, e.g. `AngularProjectedNormal`).
    latent_space::String = "flat"
    use_reactant::Bool = false
    # Reactant backend, "cpu" or "gpu"; `nothing` keeps Reactant's default (a GPU when one
    # is available).
    device::Union{Nothing, String} = nothing
    benchmark::Bool = true
    # On the Reactant path, check the device log-density+gradient against the CPU/Enzyme
    # reference at a prior draw before optimizing, and error on mismatch (catches broken
    # device models that silently produce garbage fits).
    verify_reactant::Bool = false
    start::Union{Nothing, String} = nothing
    # Reactant checkpointing (FITS + PNG + residuals): render every `sample_checkpoint`
    # samples (this is also the sampling DiskStore stride). Warmup runs in chunks of the same
    # stride, rendering the current draw each chunk, and is checkpointed every
    # `warmup_checkpoint` steps (so warmup is resumable via `restart`). 0 disables it.
    # Optimization is deliberately NOT checkpointed — rendering on the host each step is far
    # slower than the device step.
    sample_checkpoint::Int = 0
    # Warmup steps between state checkpoints on the Reactant path (`checkpoint_every` of
    # `sample`); `nothing` keeps Comrade's default.
    warmup_checkpoint::Union{Nothing, Int} = nothing
end

"""
    build_fitting_config(cfg::AbstractDict) -> FittingStrategy

Parse a fitting-strategy TOML into a [`FittingStrategy`](@ref). Sections: `[optimizer]`,
`[tempering]`, `[sampler]`, `[run]`. `sampler.algorithm` is `"nuts"` (default), whose backend
(AdvancedHMC NUTS vs Reactant NUTS with its diagonal metric) follows `run.use_reactant`, or
`"mclmc"` or `"walnuts"`, which run `sampler.chains` chains of KineticSamplers MCLMC or WALNUTS
(see [`sample_kinetic`](@ref)) and need `run.use_reactant = true`. `nadapt` is the warmup length per chain, in steps for
MCLMC (default 150 000) and in draws for WALNUTS (default 1000), and `nsample` the draws kept
per chain (default 1000). For MCLMC a draw is kept every `thin` (default 1) blocks of
`chunk_size` steps, so sampling runs `nsample * thin * chunk_size` steps per chain, and `max_step_halvings` (default 3) is the number of halvings
of a step whose energy error is too large: above `max_energy_error` in total (default 0.25 √d,
KineticSamplers' default). `integrator` is MCLMC's integrator: `"mclachlan"` (default),
`"leapfrog"` or `"baoab"` (see `KineticSamplers.BAOAB`). `max_mode_phase` is the phase per step
at the median curvature of the MCLMC warmup's step-size cap (default 0.7 with McLachlan; it must
be set with the other integrators; see [`sample_kinetic`](@ref)). `final` (default 20% of
`nadapt`) is the MCLMC steps per chain of the warmup's final phase, which runs after the
trajectory length is set, at the capped step size and that trajectory length, so the chains
settle at the sampling parameters before any draw is kept. `stiff_directions` (default 8) and
`stiff_draws` (default 2) set the `KineticSamplers.StiffDirections` of the inverse mass of MCLMC
and WALNUTS: the number of stiff directions it shrinks and the draws whose Hessians each
estimate uses. For MCLMC, `trajectory_rows` (default `"all"`) is the rows of the warmup's windows
whose autocorrelation times set the trajectory length: `"all"` the recorded coordinates and the
projections onto the low-rank directions of the inverse mass, `"projections"` the projections
alone, which include the slowest directions. `trajectory_quantile` (default: the mean) takes
that quantile of their autocorrelation times instead, `refit_trajectory_step` (default 1) is the
fraction of the way, on a log scale, that each window after a refit moves the trajectory length
toward its estimate, and `window_iterations` (default 1) is the number of windows after the
adaptation, each starting at the trajectory length the one before gave. `trajectory_length`, in
units of √d, fixes MCLMC's trajectory length, and the warmup then runs no window; left out, the
windows set it. `step_size` fixes MCLMC's sampling step size; left out, the warmup's adapted
step size is used. `bias_tuning` (default true unless `step_size` is set) reduces the warmup's
MCLMC step size until a coupled bias check passes (see [`sample_kinetic`](@ref));
`bias_tuning = true` is an error with `step_size`. After sampling, the run logs and writes the
autocorrelation times of the scalar parameters in units of the trajectory length (see
[`trajectory_report`](@ref)). For WALNUTS every draw is kept, `chunk_size`
(default 10) is the draws per sampling block, which must divide `nsample`, and `max_tree_depth`
(default 10) is the most trajectory doublings. `step_size` is an error with WALNUTS, the run
checkpoints are errors with either algorithm, and `max_tree_depth` is an error with MCLMC. `run.start` may also name the
run base of an MCLMC or WALNUTS run (the directory holding its `chain_k`), whose chains' last
draws start the chains one to one.

`run.device` (`"cpu"` or `"gpu"`) selects the Reactant backend and needs
`run.use_reactant = true`. Left out, Reactant uses a GPU when one is available and the CPU
otherwise. `"gpu"` is an error when no GPU is available.

`[optimizer] fix_scales` lists scalar sky parameters (e.g. `["σb", "σc", "σd"]`) that every
optimization stage holds at the median of their prior; the other parameters are optimized
as usual. After the last stage, every non-centered sky field `X` with scale `σX` (in
`polexp_srf`: `a`, `b`, `c`, `d`) is rescaled to `σX * rms(X)` and `X / rms(X)`, which
leaves the image unchanged and puts the white coefficients at the radius the prior's typical
set has (`rms(X) = 1`); see [`rescale_fields`](@ref). The sampler starts from that point
with every parameter free. Empty (the default) changes nothing. A name that is not a scalar
sky parameter of the model is an error when optimization starts.

`[[sampler.moves]]` tables list Metropolis–Hastings moves run between the Reactant NUTS
chunks (see [`build_moves`](@ref)), each table one move kind:

```toml
[[sampler.moves]]
kind = "mean_field"
params = ["fwhm"]     # optional: a subset of the parameters the kind acts on
rounds = 10           # optional: proposals of each move per call (default 1)
target_accept = 0.45  # optional: warmup acceptance target (default 0.45)
initial_scale = 0.01  # optional: initial random-walk step scale (default per kind)
band_limit = 3.0      # optional, field_scale and rho_field only (default inf), see below
```

The kinds, all of which leave the likelihood unchanged unless `band_limit` is finite:

  - `"flux_gain"`: total flux against a common shift of the gain log-amplitudes, `lg1μ` when
    the gain scheme has it and the `lg1` chain otherwise (no `params`);
  - `"field_scale"`: each non-centered sky field against its scale `σX` (`params`: fields);
  - `"rho_field"`: each spectral parameter of each stationary random field (Markov RF
    correlation lengths, Matérn outer scale and slope) against the field's white
    coefficients (`params`: fields);
    for these two kinds a finite `band_limit` compensates only the white coefficients with
    wavenumber up to `band_limit` times the longest baseline (stationary random fields
    only); the image then changes at higher wavenumbers and the move is accepted on the
    full posterior;
  - `"mean_field"`: each mean-model parameter against the log-intensity field `a` (PolExp
    stationary random-field models; `params`: mean-model parameters);
  - `"phase_sheet"`: a `±2π` shift of a real-line Gauss–Markov phase chain from a point on
    (`params`: chain terms; no `initial_scale`); `"stdnormal"` space only;
  - `"chain_hyper"`: each fitted hyperparameter field (`σ`, `τ`, `D`) of a Gauss–Markov
    instrument chain against the chain's whitened innovations, with the chain values fixed;
    one move per term and field, which steps every site's hyperparameter and accepts each
    site on its own, so a round is one sweep over the sites (`params`: chain terms);
    `"stdnormal"` space only.

A kind may appear more than once only with disjoint `params`. Moves run with the Reactant
NUTS, MCLMC and WALNUTS samplers (`use_reactant = true`), not with the CPU NUTS sampler.
Unknown kinds and keys are errors, as are the list form `moves = [...]` and
`moves_per_chunk`. With NUTS the moves run once per sampler chunk
(`chunk_size`, or `sample_checkpoint` when set); with MCLMC on every chain once per 50 blocks
of `chunk_size` steps, and with WALNUTS on every chain once per 10 blocks.

`[run] latent_space` picks the space the optimizer and sampler work in: `"flat"` (default,
`asflat`) or `"stdnormal"` (the `StdNormal` transport, where the prior is exactly N(0, I);
required by priors without a flat transform such as `AngularProjectedNormal`). In the
`"stdnormal"` space the start point's real-line phase chains are re-wrapped to their shortest
steps (see [`unwrap_phase_chains`](@ref)) before sampling.
"""
function build_fitting_config(cfg::AbstractDict)
    check_config_keys(
        cfg, ("optimizer", "tempering", "sampler", "run"),
        "the fitting config (top level)"
    )
    opt = get(cfg, "optimizer", Dict{String, Any}())
    temp = get(cfg, "tempering", Dict{String, Any}())
    samp = get(cfg, "sampler", Dict{String, Any}())
    run = get(cfg, "run", Dict{String, Any}())

    check_config_keys(
        opt, ("method", "maxiters", "ntrials", "g_tol", "eta", "fix_scales"), "[optimizer]"
    )
    check_config_keys(temp, ("noise_schedule",), "[tempering]")
    check_config_keys(
        samp,
        (
            "nsample", "nadapt", "step_size", "target_accept", "init_buffer",
            "term_buffer", "max_tree_depth", "chunk_size", "moves", "moves_per_chunk",
            "algorithm", "chains", "max_step_halvings", "integrator", "max_energy_error",
            "max_mode_phase", "trajectory_length", "bias_tuning", "thin", "final",
            "stiff_directions", "stiff_draws", "trajectory_rows", "trajectory_quantile",
            "refit_trajectory_step", "window_iterations",
        ),
        "[sampler]"
    )
    check_config_keys(
        run,
        (
            "use_reactant", "benchmark", "verify_reactant", "start", "sample_checkpoint",
            "checkpoint", "latent_space", "warmup_checkpoint", "device",
        ),
        "[run]"
    )

    use_reactant = Bool(get(run, "use_reactant", false))
    ahmc_only = filter(k -> haskey(samp, k), ["target_accept", "init_buffer", "term_buffer"])
    (use_reactant && !isempty(ahmc_only)) && error(
        "sampler.$(join(ahmc_only, ", sampler.")) apply to AdvancedHMC NUTS only " *
            "(run.use_reactant = false); Reactant NUTS has no setting for them"
    )
    device = haskey(run, "device") ? String(run["device"]) : nothing
    isnothing(device) || device in ("cpu", "gpu") ||
        error("unknown run.device '$device'. Allowed: cpu, gpu")
    (isnothing(device) || use_reactant) ||
        error("run.device needs run.use_reactant = true: without Reactant everything runs on the CPU")
    latent_space = String(get(run, "latent_space", "flat"))
    latent_space in ("flat", "stdnormal") ||
        error("unknown run.latent_space '$latent_space'. Allowed: flat, stdnormal")

    opt_method = String(get(opt, "method", "Adam"))
    opt_method in ("Adam", "AdamW", "LBFGS") ||
        error("unknown optimizer.method '$opt_method'. Allowed: Adam, AdamW, LBFGS")

    startval = get(run, "start", "")
    start = (startval == "") ? nothing : String(startval)

    fix_scales = get(opt, "fix_scales", String[])
    (fix_scales isa AbstractVector && all(v -> v isa AbstractString, fix_scales)) ||
        error("optimizer.fix_scales must be a list of sky parameter names, got $(repr(fix_scales))")
    allunique(fix_scales) || error("optimizer.fix_scales lists a name twice: $fix_scales")

    haskey(samp, "moves_per_chunk") && error(
        "sampler.moves_per_chunk is not a fitting-config key; set `rounds` in each [[sampler.moves]] table"
    )
    moves = _parse_move_specs(get(samp, "moves", Any[]), latent_space)
    isempty(moves) || use_reactant || error(
        "sampler.moves needs run.use_reactant = true: the CPU NUTS sampler does not run moves"
    )

    algorithm = String(get(samp, "algorithm", "nuts"))
    algorithm in ("nuts", KINETIC_ALGORITHMS...) || error(
        "sampler.algorithm must be \"nuts\", \"mclmc\" or \"walnuts\", got $(repr(algorithm))"
    )
    kinetic = algorithm in KINETIC_ALGORITHMS
    for (key, allowed) in (
            "step_size" => ("nuts", "mclmc"), "max_tree_depth" => ("nuts", "walnuts"),
            "chains" => KINETIC_ALGORITHMS, "max_step_halvings" => ("mclmc",),
            "integrator" => ("mclmc",), "max_energy_error" => ("mclmc",),
            "max_mode_phase" => ("mclmc",),
            "trajectory_length" => ("mclmc",), "bias_tuning" => ("mclmc",), "thin" => ("mclmc",),
            "final" => ("mclmc",), "trajectory_rows" => ("mclmc",),
            "trajectory_quantile" => ("mclmc",), "refit_trajectory_step" => ("mclmc",),
            "window_iterations" => ("mclmc",),
            "stiff_directions" => KINETIC_ALGORITHMS, "stiff_draws" => KINETIC_ALGORITHMS,
        )
        haskey(samp, key) && !(algorithm in allowed) && error(
            "sampler.$key applies to sampler.algorithm = $(join(repr.(allowed), " or ")) only, " *
                "not $(repr(algorithm))"
        )
    end
    if kinetic
        use_reactant || error("sampler.algorithm = \"$algorithm\" needs run.use_reactant = true")
        run_nuts_only = filter(
            k -> get(run, k, 0) != 0, ["sample_checkpoint", "checkpoint", "warmup_checkpoint"]
        )
        isempty(run_nuts_only) ||
            error("run.$(join(run_nuts_only, ", run.")) apply to NUTS only")
    end
    chains = Int(get(samp, "chains", 4))
    chains >= 1 || error("sampler.chains must be at least 1, got $chains")
    nsample = Int(get(samp, "nsample", kinetic ? 1_000 : 10_000))
    chunk_size = Int(get(samp, "chunk_size", algorithm == "walnuts" ? 10 : 100))
    chunk_size >= 1 || error("sampler.chunk_size must be at least 1, got $chunk_size")
    thin = Int(get(samp, "thin", 1))
    thin >= 1 || error("sampler.thin must be at least 1, got $thin")
    nadapt = Int(get(samp, "nadapt", get(KINETIC_WARMUP, algorithm, 5_000)))
    final = Int(get(samp, "final", algorithm == "mclmc" ? round(Int, 0.2 * nadapt) : 0))
    final >= 0 || error("sampler.final must be nonnegative, got $final")
    stiff_directions = Int(get(samp, "stiff_directions", 8))
    stiff_directions >= 1 ||
        error("sampler.stiff_directions must be at least 1, got $stiff_directions")
    stiff_draws = Int(get(samp, "stiff_draws", 2))
    stiff_draws >= 1 || error("sampler.stiff_draws must be at least 1, got $stiff_draws")
    trajectory_rows = String(get(samp, "trajectory_rows", "all"))
    trajectory_rows in ("all", "projections") || error(
        "sampler.trajectory_rows must be \"all\" or \"projections\", got $(repr(trajectory_rows))"
    )
    trajectory_quantile = haskey(samp, "trajectory_quantile") ?
        Float64(samp["trajectory_quantile"]) : nothing
    isnothing(trajectory_quantile) || 0 <= trajectory_quantile <= 1 ||
        error("sampler.trajectory_quantile must be in [0, 1], got $trajectory_quantile")
    refit_trajectory_step = Float64(get(samp, "refit_trajectory_step", 1.0))
    0 < refit_trajectory_step <= 1 ||
        error("sampler.refit_trajectory_step must be in (0, 1], got $refit_trajectory_step")
    window_iterations = Int(get(samp, "window_iterations", 1))
    window_iterations >= 1 ||
        error("sampler.window_iterations must be at least 1, got $window_iterations")
    (algorithm == "walnuts" && nsample % chunk_size != 0) && error(
        "sampler.nsample = $nsample must be a multiple of sampler.chunk_size = $chunk_size, " *
            "the WALNUTS draws per block"
    )
    integrator = _mclmc_integrator_name(get(samp, "integrator", "mclachlan"))
    algorithm == "mclmc" && integrator != "mclachlan" && !haskey(samp, "max_mode_phase") && error(
        "sampler.max_mode_phase must be set with sampler.integrator = $(repr(integrator)): " *
            "the default phase of the MCLMC step-size cap applies to \"mclachlan\""
    )
    bias_tuning = Bool(get(samp, "bias_tuning", !haskey(samp, "step_size")))
    bias_tuning && haskey(samp, "step_size") && error(
        "sampler.step_size and sampler.bias_tuning = true both set the MCLMC step size; set one"
    )
    return FittingStrategy(;
        opt_method = opt_method,
        maxiters = Int(get(opt, "maxiters", 10_000)),
        ntrials = Int(get(opt, "ntrials", 5)),
        g_tol = Float64(get(opt, "g_tol", 0.1)),
        eta = Float64(get(opt, "eta", 0.001)),
        fix_scales = Symbol.(fix_scales),
        noise_schedule = Float64.(get(temp, "noise_schedule", [0.05, 0.025, 0.0])),
        nsample,
        nadapt,
        step_size = algorithm == "mclmc" ?
            _positive_or_nothing(get(samp, "step_size", nothing), "sampler.step_size") :
            Float64(get(samp, "step_size", 0.01)),
        target_accept = Float64(get(samp, "target_accept", 0.9)),
        init_buffer = Int(get(samp, "init_buffer", 200)),
        term_buffer = Int(get(samp, "term_buffer", 500)),
        max_tree_depth = Int(get(samp, "max_tree_depth", 10)),
        chunk_size,
        thin,
        final,
        stiff_directions,
        stiff_draws,
        trajectory_rows,
        trajectory_quantile,
        refit_trajectory_step,
        window_iterations,
        moves,
        algorithm,
        chains,
        max_step_halvings = Int(get(samp, "max_step_halvings", 3)),
        integrator,
        max_energy_error = _positive_or_nothing(
            get(samp, "max_energy_error", nothing), "sampler.max_energy_error"
        ),
        max_mode_phase = _positive_or_nothing(
            get(samp, "max_mode_phase", nothing), "sampler.max_mode_phase"
        ),
        trajectory_length = _positive_or_nothing(
            get(samp, "trajectory_length", nothing), "sampler.trajectory_length"
        ),
        bias_tuning,
        latent_space = latent_space,
        use_reactant = use_reactant,
        device,
        benchmark = Bool(get(run, "benchmark", true)),
        verify_reactant = Bool(get(run, "verify_reactant", false)),
        start = start,
        # `checkpoint` is accepted as an alias for `sample_checkpoint`.
        sample_checkpoint = Int(get(run, "sample_checkpoint", get(run, "checkpoint", 0))),
        warmup_checkpoint = haskey(run, "warmup_checkpoint") ? Int(run["warmup_checkpoint"]) : nothing,
    )
end

# The `space` argument of `Comrade.maybe_transport` for a strategy: `nothing` is the flat space.
latent_space(strategy::FittingStrategy) = strategy.latent_space == "stdnormal" ? PT.StdNormal() : nothing

const _MOVE_KEYS = ("kind", "params", "rounds", "target_accept", "initial_scale", "band_limit")

function _parse_move_specs(tables, latent_space)
    (tables isa AbstractVector && all(t -> t isa AbstractDict, tables)) || error(
        "sampler.moves must be [[sampler.moves]] tables, each naming a move kind, e.g.\n" *
            "[[sampler.moves]]\nkind = \"mean_field\"\nrounds = 10\ngot $(repr(tables))"
    )
    specs = map(enumerate(tables)) do (i, t)
        where_ = "[[sampler.moves]] table $i"
        check_config_keys(t, _MOVE_KEYS, where_)
        kind = get(t, "kind", nothing)
        kind isa AbstractString || error("$where_ needs a kind (one of $(collect(MOVE_KINDS)))")
        kind in MOVE_KINDS || error("unknown move kind \"$kind\" in $where_. Allowed: $(collect(MOVE_KINDS))")
        params = get(t, "params", nothing)
        isnothing(params) || (params isa AbstractVector && !isempty(params) && all(p -> p isa AbstractString, params)) ||
            error("$where_: params must be a non-empty list of names, got $(repr(params))")
        rounds = get(t, "rounds", 1)
        (rounds isa Integer && rounds >= 1) || error("$where_: rounds must be an integer ≥ 1, got $(repr(rounds))")
        target_accept = get(t, "target_accept", 0.45)
        (target_accept isa Real && 0 < target_accept < 1) ||
            error("$where_: target_accept must lie in (0, 1), got $(repr(target_accept))")
        initial_scale = get(t, "initial_scale", nothing)
        isnothing(initial_scale) || (initial_scale isa Real && initial_scale > 0) ||
            error("$where_: initial_scale must be positive, got $(repr(initial_scale))")
        (kind == "phase_sheet" && !isnothing(initial_scale)) &&
            error("$where_: phase_sheet takes discrete ±2π steps and has no initial_scale")
        (kind == "flux_gain" && !isnothing(params)) && error("$where_: flux_gain takes no params")
        band_limit = get(t, "band_limit", Inf)
        (band_limit isa Real && band_limit > 0) || error(
            "$where_: band_limit must be a positive multiple of the longest baseline (inf for no limit), got $(repr(band_limit))"
        )
        (isfinite(band_limit) && !(kind in ("field_scale", "rho_field"))) &&
            error("$where_: band_limit applies only to field_scale and rho_field")
        (kind == "phase_sheet" && latent_space != "stdnormal") && error(
            "$where_: phase_sheet needs run.latent_space = \"stdnormal\" (the flat space's wrapped chains have no sheets)"
        )
        (kind == "chain_hyper" && latent_space != "stdnormal") && error(
            "$where_: chain_hyper needs run.latent_space = \"stdnormal\" (its sites are accepted without the likelihood)"
        )
        MoveSpec(;
            kind = String(kind), params = isnothing(params) ? nothing : String.(params),
            rounds = Int(rounds), target_accept = Float64(target_accept),
            initial_scale = isnothing(initial_scale) ? nothing : Float64(initial_scale),
            band_limit = Float64(band_limit),
        )
    end
    for kind in unique(s.kind for s in specs)
        same = [s.params for s in specs if s.kind == kind]
        length(same) == 1 && continue
        (any(isnothing, same) || !allunique(reduce(vcat, same))) && error(
            "sampler.moves lists kind \"$kind\" more than once with overlapping params; give each table disjoint params"
        )
    end
    return MoveSpec[specs...]
end
