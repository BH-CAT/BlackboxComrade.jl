# KineticSamplers MCLMC and WALNUTS chains in the run's latent space.

const KINETIC_START_JITTER = 0.001
const KINETIC_DRAWS_PER_FILE = 10
const KINETIC_RETURNED_DRAWS = 500
const KINETIC_LOG_DRAWS = 50
const KINETIC_READ_DRAWS = 200
const MCLMC_MOVE_BLOCKS = 50
const WALNUTS_MOVE_BLOCKS = 10
const MCLMC_CHECK_BLOCKS = 10
const SCALAR_BLOCK = 8
const TRAJECTORY_REPORTED = 5
const CAP_QUANTILE = 0.99
const MCLACHLAN_CAP_PHASE = 1.2
const BIAS_ESS = 100
const BIAS_STEPS = 2000
const BIAS_MAX_STEPS = 16 * BIAS_STEPS
const BIAS_MAX_ROUNDS = 10
const BIAS_BUDGET = 0.25
const FINAL_TREND_BLOCKS = 5

"""
    LatentTarget(tpost, dim)

The LogDensityProblems interface of the device posterior `tpost` in its latent space, with
gradients and Hessian-vector products by Enzyme under Reactant.
"""
struct LatentTarget{P}
    tpost::P
    dim::Int
end

LogDensityProblems.capabilities(::Type{<:LatentTarget}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(t::LatentTarget) = t.dim
LogDensityProblems.logdensity(t::LatentTarget, x) = Comrade.logdensityof(t.tpost, x)

# `set_strong_zero` sends 0*Inf and 0*NaN to 0 in the derivative passes, without which a stiff
# image model has no finite gradient.
function LogDensityProblems.logdensity_and_gradient(t::LatentTarget, x)
    mode = Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal)
    derivs, val = Enzyme.gradient(mode, Comrade.logdensityof, Enzyme.Const(t.tpost), x)
    return val, last(derivs)
end

_latent_gradient(tpost, x) = last(LogDensityProblems.logdensity_and_gradient(LatentTarget(tpost, length(x)), x))

KineticSamplers.has_hessian_vector_product(::Type{<:LatentTarget}) = true

# Forward-mode differentiation of the reverse-mode gradient along `v`.
function KineticSamplers.logdensity_and_hessian_vector_product(t::LatentTarget, x, v)
    hv = only(
        Enzyme.autodiff(
            Enzyme.set_strong_zero(Enzyme.Forward), Enzyme.Const(_latent_gradient),
            Enzyme.Const(t.tpost), Enzyme.Duplicated(x, v)
        )
    )
    return Comrade.logdensityof(t.tpost, x), hv
end

# Device buffers are freed only when Julia collects their wrappers, which it does not do on
# device memory pressure.
collect_device_garbage(block) = block % 50 == 0 ? GC.gc() : (block % 5 == 0 && GC.gc(false))

"""
    sample_kinetic(out, post, xopt, strategy, rng; starts=nothing, resume=false) -> Comrade.PosteriorSamples

`strategy.chains` chains of the KineticSamplers algorithm `strategy.algorithm` (MCLMC or
WALNUTS) on `post` in the latent space of `strategy.latent_space`, compiled with Reactant, started at `xopt`
jittered by `$(KINETIC_START_JITTER)` standard normals per latent coordinate, or chain `k` at
`starts[k]` when `starts` is given. A `PooledAdaptation` of `strategy.nadapt` MCLMC steps or
WALNUTS draws per chain, in blocks of `strategy.chunk_size` MCLMC steps or of KineticSamplers'
default length for WALNUTS, adapts one step size and one low-rank inverse mass with
`strategy.stiff_directions` stiff directions, each estimate from `strategy.stiff_draws` draws,
for all chains. The MCLMC step size follows its energy error and is capped at the step size for
which a harmonic mode of the $(CAP_QUANTILE) quantile of the whitened curvature at positions of
the warmup's records advances by the phase [`mclmc_cap_phase`](@ref) per step.
The MCLMC warmup sets its trajectory length from the autocorrelation times of the rows
`strategy.trajectory_rows`, by their mean or their `strategy.trajectory_quantile` quantile, moves
it by `strategy.refit_trajectory_step` after each refit, runs `strategy.window_iterations`
windows after the adaptation, and ends with a final phase of `strategy.final` steps per chain.
MCLMC warms up without step halving, except with `integrator = "baoab"`, whose warmup halves
steps as its sampling does. Its history and parameters go to `out/mclmc_warmup.jls` or
`out/walnuts_warmup.jls`. While it runs, `out/warmup` holds a `WarmupRecord`: each chain's log
density after every block in `out/warmup/trace.jls`, rewritten at each logged block (every
$(KINETIC_LOG_DRAWS) draws' worth of blocks), and its position at each logged block as a draw of
`out/warmup/chain_k`. After a final phase, a warning names the scalar parameters that still
trend in it (see [`warn_trends`](@ref)). With `strategy.trajectory_length`, the MCLMC warmup runs
no window and the trajectory length is `strategy.trajectory_length` times √d, and with
`strategy.step_size` MCLMC samples at that step size instead of the warmup's. After the MCLMC
warmup, `KineticSamplers.measure_bias` estimates the bias of the sampling parameters from
coupled runs of $(BIAS_STEPS) steps with one and two substeps (see [`bias_check`](@ref)), whose
tolerances are the Monte Carlo errors at an effective sample size of $(BIAS_ESS). With
`strategy.bias_tuning`, `KineticSamplers.tune_step_size` reduces the step size until the
estimates are within them, in at most $(BIAS_MAX_ROUNDS) rounds of coupled runs of up to
$(BIAS_MAX_STEPS) steps and at most $(BIAS_BUDGET) of the sampling's gradients (see
[`bias_budget`](@ref)), logging each round, and the warmup file holds the reduced step size.
Without it the estimate is logged, with a warning if some estimate exceeds its tolerance by more
than its margin of standard errors, and sampling continues at the warmup's step size. Then $(MCLMC_CHECK_BLOCKS) blocks at the sampling parameters
follow, whose energy statistics are logged per chain; if they fail
`KineticSamplers.check_energy`, a warning says so and sampling continues.

With `resume`, MCLMC sampling whose warmup finished (`out/mclmc_warmup.jls`) and which wrote
draws continues instead: no warmup, the parameters of that file, each chain from its last draw,
appending draws until `strategy.nsample`, without the bias check or tuning; the trajectory report then
covers all draws. Without draws it starts afresh, which needs `out/warmup` moved away.

Sampling keeps `strategy.nsample` draws per chain, chain `k` written as a Comrade sample
directory `out/chain_k`: for MCLMC the last position of every `strategy.thin`-th block, so the
chains run `strategy.nsample * strategy.thin` blocks and each kept draw's `num_grads`, `time`,
`divergent` and `halved` count all the blocks since the previous kept draw; every draw for WALNUTS
(`WALNUTS(; max_depth = strategy.max_tree_depth)`). The moves of `strategy` run on every chain
in every phase of the warmup and in sampling, every $(MCLMC_MOVE_BLOCKS) blocks for MCLMC
and every $(WALNUTS_MOVE_BLOCKS) blocks for WALNUTS; a chain that moved restarts at its new position. Returns about
$(KINETIC_RETURNED_DRAWS) draws, evenly thinned from each chain, as a draws × chains
`PosteriorSamples`; the chains on disk keep every draw (see [`load_kinetic_chains`](@ref)).
After MCLMC sampling, [`trajectory_report`](@ref) logs the autocorrelation times of the scalar
parameters and writes them to `out/trajectory_report.toml`.
"""
function sample_kinetic(out, post, xopt, strategy::FittingStrategy, rng; starts = nothing, resume = false)
    K = strategy.chains
    chain_dirs = [joinpath(out, "chain_$k") for k in 1:K]
    existing = filter(ispath, chain_dirs)
    resume || isempty(existing) || error(
        "$(first(existing)) exists; restart the run to continue its sampling, or move the earlier chains away"
    )
    post_cpu = @set post.admode = nothing
    space = latent_space(strategy)
    tpost = Comrade.maybe_transport(
        Comrade.prepare_device(post_cpu, Comrade.ComradeBase.ReactantEx()), space
    )
    tb = Comrade.CoordinateView(post_cpu, space).tbase
    d = dimension(tb)
    check = bias_check()
    target = KineticSamplers.Target(LatentTarget(tpost, d); backend = ADTypes.AutoReactant())
    alg, warmup_alg = kinetic_samplers(strategy, d)
    name = String(nameof(typeof(alg)))
    warmup_file = joinpath(out, "$(lowercase(name))_warmup.jls")
    done = resume ? _written_draws(chain_dirs) : 0
    done > 0 && !isfile(warmup_file) &&
        error("$out holds $done draws per chain but no $(basename(warmup_file)); sampling cannot resume")
    done > 0 && alg isa KineticSamplers.WALNUTS && error("resuming WALNUTS sampling is not supported")
    done > strategy.nsample &&
        error("$out holds $done draws per chain, more than sampler.nsample = $(strategy.nsample)")
    nadapt, block = strategy.nadapt, strategy.chunk_size
    layout = block_layout(alg, block)
    strategy.nsample % layout.draws == 0 || error(
        "$(strategy.nsample) draws per chain are not a whole number of blocks of $(layout.draws)"
    )
    log_blocks = cld(KINETIC_LOG_DRAWS, layout.draws)
    stiff = KineticSamplers.StiffDirections(;
        directions = strategy.stiff_directions, draws = strategy.stiff_draws
    )
    mass = KineticSamplers.HistoryMassAdaptation(;
        max_rank = 100, max_wide = 80, stiff, vecs_eltype = Float32
    )
    adapt = if alg isa KineticSamplers.WALNUTS
        KineticSamplers.PooledAdaptation(nadapt; mass)
    else
        KineticSamplers.PooledAdaptation(
            nadapt; mass, block, window = isnothing(strategy.trajectory_length) ? nothing : 0,
            final = strategy.final, curvature_quantile = CAP_QUANTILE,
            max_mode_phase = mclmc_cap_phase(alg, strategy),
            trajectory_rows = Symbol(strategy.trajectory_rows),
            strategy.trajectory_quantile, strategy.refit_trajectory_step,
            strategy.window_iterations,
        )
    end
    warmup_block = KineticSamplers.block_length(adapt, warmup_alg)
    Θ = if isnothing(starts)
        Comrade.inverse(tb, xopt) .+ KINETIC_START_JITTER .* randn(rng, d, K)
    else
        length(starts) == K ||
            error("$(length(starts)) starting points for sampler.chains = $K chains")
        reduce(hcat, [Comrade.inverse(tb, s) for s in starts])
    end
    host_rngs(n) = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:n]
    restart_rngs = host_rngs(K)
    rngs = KineticSamplers.backend_rngs(ADTypes.AutoReactant(), host_rngs(K))

    moves = isempty(strategy.moves) ? nothing : build_moves(
            post_cpu, strategy.moves, xopt; space, output = joinpath(mkpath(out), "moves.jls")
        )
    function move_chains(chains, phase, step, total)
        Θh = Array(chains.state.θ)
        moved = false
        for k in axes(Θh, 2)
            x = Reactant.to_rarray(Θh[:, k])
            x′ = moves(x, tpost, phase, rng; step, total, report = false)
            if x′ !== x
                Θh[:, k] = Array(x′)
                moved = true
            end
        end
        GC.gc()
        moved || return chains
        return KineticSamplers.init_chains(
            restart_rngs, target, phase === :warmup ? warmup_alg : alg, Θh
        )
    end

    if done > 0
        host_params = deserialize(warmup_file).params
        m = host_params.inverse_mass
        length(m.stds) == d ||
            error("$(basename(warmup_file)) has $(length(m.stds)) latent coordinates, the posterior $d")
        params = KineticSamplers.IsokineticParameters(
            host_params.step_size, host_params.trajectory_length,
            KineticSamplers.LowRankInverseMass(m.stds, m.vecs, m.vals)
        )
        chains = KineticSamplers.init_chains(
            restart_rngs, target, alg, reduce(hcat, [Comrade.inverse(tb, s) for s in _last_draws(out)])
        )
        @info @sprintf(
            "%s sampling resumes after draw %d of %d per chain, at step size %.4g and trajectory length %.4g from %s",
            name, done, strategy.nsample, host_params.step_size, host_params.trajectory_length, warmup_file
        )
    else
        ispath(joinpath(out, "warmup")) && error(
            "$(joinpath(out, "warmup")) exists; the warmup starts afresh, so move the earlier warmup record away"
        )
        @info "$name: $K chains on $d latent coordinates, warmup $nadapt $(layout.unit) per chain in " *
            "blocks of $warmup_block, $(strategy.nsample) draws per chain" *
            (strategy.thin > 1 ? ", one every $(strategy.thin) blocks" : "") * "; $(repr(alg))"
        record = WarmupRecord(joinpath(out, "warmup"), K)
        final_scalars = Vector{Vector{Pair{String, Float64}}}[]
        t0 = time()
        function between_blocks(chains, info)
            collect_device_garbage(info.block)
            logp = Array(chains.state.logp)
            record_block!(record, info, time() - t0, logp)
            if info.block % log_blocks == 0
                Θh = Array(chains.state.θ)
                record_positions!(record, info, logp, [Comrade.transform(tb, Θh[:, k]) for k in axes(Θh, 2)])
                write_trace(record)
                @info @sprintf(
                    "%s warmup, %s block %d: %d %s per chain, log density %s (%.0f s)",
                    name, info.phase, info.block, info.steps, layout.unit,
                    join((@sprintf("%.0f", l) for l in logp), " "), time() - t0
                )
            end
            if info.phase === :final && info.block % FINAL_TREND_BLOCKS == 0
                Θh = Array(chains.state.θ)
                push!(final_scalars, [scalar_leaves(Comrade.transform(tb, Θh[:, k])) for k in axes(Θh, 2)])
            end
            if !isnothing(moves) && info.block % layout.move_blocks == 0
                total = info.phase === :adaptation ? nadapt : nothing
                chains = move_chains(chains, :warmup, info.steps, total)
            end
            _flush_logs()
            return chains
        end
        chains, params, history = KineticSamplers.warmup(
            host_rngs(K), target, adapt, warmup_alg, Θ; between_blocks, check = false
        )
        write_trace(record)
        foreach(flush_draws!, record.writers)
        if alg isa KineticSamplers.MCLMC
            h = something(strategy.step_size, params.step_size)
            L = isnothing(strategy.trajectory_length) ? params.trajectory_length :
                strategy.trajectory_length * sqrt(d)
            isnothing(strategy.step_size) || @info @sprintf(
                "MCLMC step size %.4g from sampler.step_size (warmup: %.4g)", h, params.step_size
            )
            isnothing(strategy.trajectory_length) ||
                @info "MCLMC trajectory length $(strategy.trajectory_length) √d from sampler.trajectory_length"
            params = KineticSamplers.IsokineticParameters(h, L, params.inverse_mass)
        end
        _log_warmup(alg, params, history, d, time() - t0)
        alg isa KineticSamplers.MCLMC && strategy.final > 0 && warn_trends(final_scalars)
        if alg isa KineticSamplers.MCLMC && strategy.bias_tuning
            params, chains = _tune_step_size(
                rngs, target, alg, params, chains, d, check, bias_budget(strategy)
            )
        end
        host_params = _host_params(params)
        serialize(warmup_file, (; history, params = host_params))
    end

    params = Reactant.to_rarray(params; track_numbers = Number)
    if alg isa KineticSamplers.MCLMC && done == 0 && !strategy.bias_tuning
        chains = _bias_diagnostic(rngs, target, alg, params, chains, d, check)
    end
    chains, trace = check_chains(target, rngs, alg, params, chains, block, d)
    writers = [ChainDrawWriter(dir) for dir in chain_dirs]
    scalars = [Vector{Float64}[] for _ in 1:K]
    scalar_names = String[]
    alg isa KineticSamplers.MCLMC && _read_scalars!(scalars, scalar_names, chain_dirs, done)
    health = nothing
    thin = strategy.thin
    nblocks = strategy.nsample * thin ÷ layout.draws
    ndraws = nblocks * layout.draws
    seen, pending = 0, nothing
    t0 = time()
    for b in (done * thin ÷ layout.draws + 1):nblocks
        tb0 = time()
        chains, trace = KineticSamplers.run_block(
            rngs, target, alg, params, chains, trace, block, layout.recorder
        )
        draws = block_draws(alg, chains, trace, time() - tb0)
        for (Θh, stats) in draws
            counts = [draw_health(alg, s) for s in stats]
            health = isnothing(health) ? counts : map((a, c) -> map(+, a, c), health, counts)
            pending = isnothing(pending) ? stats : map(merge_draw_stats, pending, stats)
            seen += 1
            seen % thin == 0 || continue
            for (k, w) in pairs(writers)
                θ = Comrade.transform(tb, Θh[:, k])
                push_draw!(w, θ; pending[k]...)
                alg isa KineticSamplers.MCLMC || continue
                leaves = scalar_leaves(θ)
                isempty(scalar_names) && append!(scalar_names, first.(leaves))
                push!(scalars[k], last.(leaves))
            end
            pending = nothing
        end
        if !isnothing(moves) && b % layout.move_blocks == 0
            chains = move_chains(chains, :sampling, b * layout.draws, ndraws)
        end
        collect_device_garbage(b)
        if b % log_blocks == 0 || b == nblocks
            @info @sprintf(
                "%s sampling %d/%d draws per chain: %s, log density %s (%.0f s)",
                name, done + seen ÷ thin, strategy.nsample, _health_text(health),
                join((@sprintf("%.0f", s.logp) for s in last(last(draws))), " "), time() - t0
            )
            _flush_logs()
        end
    end
    foreach(flush_draws!, writers)
    if alg isa KineticSamplers.MCLMC && !isempty(scalar_names)
        X = stack(reduce(hcat, s) for s in scalars)
        trajectory_report(
            joinpath(out, "trajectory_report.toml"), scalar_names, X, host_params.step_size,
            host_params.trajectory_length, thin * block, d
        )
    end
    isnothing(moves) || foreach(move_summary(moves)) do m
        @info "move $(m.name): warmup $(m.warmup.accepted)/$(m.warmup.proposed), " *
            "sampling $(m.sampling.accepted)/$(m.sampling.proposed) accepted" * _scale_text(m.τ)
    end
    stride = max(1, strategy.nsample ÷ cld(KINETIC_RETURNED_DRAWS, K))
    return load_kinetic_chains(out, 1:stride:strategy.nsample)
end

"""
    scalar_leaves(θ)

The scalar parameters of the draw `θ` as `name => value` pairs: its real numbers and its
tuples and vectors of at most $(SCALAR_BLOCK) reals, named by their path, such as `"sky.ρa[2]"`.
"""
scalar_leaves(θ) = scalar_leaves!(Pair{String, Float64}[], "", θ)

function scalar_leaves!(out, prefix, x)
    if x isa Real
        push!(out, prefix => Float64(x))
    elseif x isa NamedTuple
        for (k, v) in pairs(x)
            scalar_leaves!(out, isempty(prefix) ? string(k) : "$prefix.$k", v)
        end
    elseif (x isa AbstractArray{<:Real} || x isa Tuple{Vararg{Real}}) && length(x) <= SCALAR_BLOCK
        for (i, v) in pairs(x)
            push!(out, "$prefix[$i]" => Float64(v))
        end
    end
    return out
end

"""
    trajectory_report(path, names, X, h, L, steps_per_draw, d)

The autocorrelation times of the scalar parameters `names` of MCLMC chains on `d` latent
coordinates at step size `h` and trajectory length `L`, from their draws `X` (parameters ×
draws × chains, one draw every `steps_per_draw` steps), estimated from all chains together
by `KineticSamplers.multichain_times`. For each parameter, `R = τ h / L` is its time τ in steps
times `h`, in units of the trajectory length. A time above a tenth of the draws per chain is a
lower bound, and so is its `R`. A large `R` does not by itself mean that a longer trajectory
would mix the parameter faster: in curved geometry a longer trajectory can mix it slower.

Logs the $(TRAJECTORY_REPORTED) parameters with the largest `R` and writes every parameter's
time, effective sample size and `R` to `path` as TOML.
"""
function trajectory_report(path, names, X, h, L, steps_per_draw, d)
    length(names) == size(X, 1) ||
        throw(DimensionMismatch("$(length(names)) names for $(size(X, 1)) parameters"))
    n, K = size(X, 2), size(X, 3)
    τ = KineticSamplers.multichain_times(X)
    rows = [
        (;
            name, tau_steps = t * steps_per_draw, ess = n * K / t, R = t * steps_per_draw * h / L,
            resolved = t < n / 10,
        )
            for (name, t) in zip(names, τ) if isfinite(t)
    ]
    isempty(rows) && (@warn "no scalar parameter varies over the draws; no trajectory report"; return nothing)
    sort!(rows; by = r -> r.R, rev = true)
    sq = sqrt(d)
    @info @sprintf(
        "MCLMC autocorrelation times at step size %.4g and trajectory length %.4g = %.3g √d (R = τ h / L): %s",
        h, L, L / sq, join(
            (
                @sprintf(
                    "%s τ %s%.0f steps, ESS %.0f, R %.2g", r.name,
                    r.resolved ? "" : "≥ ", r.tau_steps, r.ess, r.R
                ) for r in first(rows, TRAJECTORY_REPORTED)
            ), "; "
        )
    )
    open(path, "w") do io
        TOML.print(
            io, Dict(
                "step_size" => h, "trajectory_length" => L, "trajectory_length_sqrt_d" => L / sq,
                "steps_per_draw" => steps_per_draw, "draws_per_chain" => n, "chains" => K,
                "parameters" => [Dict(String(k) => v for (k, v) in pairs(r)) for r in rows],
            )
        )
    end
    return nothing
end

const MCLMC_INTEGRATOR = Dict(
    "mclachlan" => KineticSamplers.McLachlan(), "leapfrog" => KineticSamplers.Leapfrog(),
    "baoab" => KineticSamplers.BAOAB(),
)

# The sampler of `strategy` on `d` latent coordinates and the one its warmup runs.
function kinetic_samplers(strategy::FittingStrategy, d)
    if strategy.algorithm == "mclmc"
        integrator = MCLMC_INTEGRATOR[strategy.integrator]
        max_error = isnothing(strategy.max_energy_error) ? 0.25 : strategy.max_energy_error / sqrt(d)
        sampling = KineticSamplers.MCLMC(; integrator, strategy.max_step_halvings, max_error)
        # McLachlan's warmup runs without halving, so the step-size adaptation sees the energy
        # error of every full step. BAOAB's lower stability limit makes unhalved steps diverge
        # repeatedly in stiff states, so its warmup halves as sampling does.
        integrator isa KineticSamplers.BAOAB && return sampling, sampling
        return sampling, KineticSamplers.MCLMC(; integrator, max_step_halvings = 0)
    elseif strategy.algorithm == "walnuts"
        alg = KineticSamplers.WALNUTS(; max_depth = strategy.max_tree_depth)
        return alg, alg
    end
    return error("sampler.algorithm = $(repr(strategy.algorithm)) is not a KineticSamplers algorithm")
end

"""
    mclmc_cap_phase(alg, strategy) -> Float64

The phase per step, at the $(CAP_QUANTILE) quantile of the whitened curvature, of the MCLMC
warmup's step-size cap:
`strategy.max_mode_phase`, the phase at which a harmonic mode's variance is within
`strategy.max_mode_bias` of the target's (`KineticSamplers.mode_phase_for_bias`), or
$(MCLACHLAN_CAP_PHASE) with McLachlan's integrator. Other integrators have no default.
"""
function mclmc_cap_phase(alg, strategy)
    isnothing(strategy.max_mode_phase) || return strategy.max_mode_phase
    isnothing(strategy.max_mode_bias) ||
        return KineticSamplers.mode_phase_for_bias(alg.integrator, strategy.max_mode_bias)
    alg.integrator isa KineticSamplers.McLachlan && return MCLACHLAN_CAP_PHASE
    return error(
        "sampler.max_mode_phase or sampler.max_mode_bias must be set for the integrator $(nameof(typeof(alg.integrator))): " *
            "the default phase of the step-size cap, $(MCLACHLAN_CAP_PHASE) at the $(CAP_QUANTILE) quantile of the curvature, applies to McLachlan's"
    )
end

# A block of `block` MCLMC steps keeps its last position; one of `block` WALNUTS draws keeps
# every draw.
block_layout(::KineticSamplers.MCLMC, block) = (;
    unit = "steps", draws = 1, move_blocks = MCLMC_MOVE_BLOCKS,
    recorder = KineticSamplers.NoPositions(),
)
block_layout(::KineticSamplers.WALNUTS, block) = (;
    unit = "draws", draws = block, move_blocks = WALNUTS_MOVE_BLOCKS, recorder = KineticSamplers.RecordPositions(),
)

# The statistics of the MCLMC blocks `a` then `b` taken together, as kept for a thinned draw.
merge_draw_stats(a, b) = (;
    logp = b.logp, num_grads = a.num_grads + b.num_grads, time = a.time + b.time,
    divergent = a.divergent + b.divergent, halved = a.halved + b.halved,
)

# The draws of a sampling block that took `seconds`, as (positions `d × K`, statistics of each
# chain) pairs.
function block_draws(::KineticSamplers.MCLMC, chains, trace, seconds)
    Θh = Array(chains.state.θ)
    logp, grads = Array(trace.logp), Array(trace.num_grads)
    diverged, halvings = Array(trace.diverged), Array(trace.step_halvings)
    stats = map(axes(Θh, 2)) do k
        (;
            logp = logp[end, k], num_grads = sum(view(grads, :, k)), time = seconds / size(Θh, 2),
            divergent = count(view(diverged, :, k)), halved = count(>(0), view(halvings, :, k)),
        )
    end
    return [(Θh, stats)]
end

function block_draws(::KineticSamplers.WALNUTS, chains, trace, seconds)
    positions = Array(trace.positions)
    logp, grads, depth = Array(trace.logp), Array(trace.num_grads), Array(trace.depth)
    accept, failed = Array(trace.accept_stat), Array(trace.leaf_failed)
    per_draw = seconds / (size(positions, 2) * size(positions, 3))
    return map(axes(positions, 2)) do j
        stats = map(axes(positions, 3)) do k
            (;
                logp = logp[j, k], num_grads = grads[j, k], time = per_draw, tree_depth = depth[j, k],
                accept_stat = accept[j, k], leaf_failed = failed[j, k],
            )
        end
        (positions[:, j, :], stats)
    end
end

draw_health(::KineticSamplers.MCLMC, s) = (; divergent_steps = s.divergent, halved_steps = s.halved)
draw_health(alg::KineticSamplers.WALNUTS, s) = (;
    failed_draws = Int(s.leaf_failed), max_depth_draws = Int(s.tree_depth == alg.max_depth),
)

_health_text(health) = join(
    ("$(replace(String(k), "_" => " ")) $(join(getfield.(health, k), " "))" for k in keys(first(health))),
    ", "
)

"""
    load_kinetic_chains(out, nsamples=:) -> Comrade.PosteriorSamples

The draws of the chains `out/chain_1`, `out/chain_2`, … of an MCLMC or WALNUTS run (see
[`sample_kinetic`](@ref)) as a draws × chains `PosteriorSamples`, with `nsamples` selecting
draws of each chain.
"""
function load_kinetic_chains(out, nsamples = Base.Colon())
    per_chain = [Comrade.load_samples(dir, nsamples) for dir in _chain_dirs(out)]
    lengths = length.(per_chain)
    allequal(lengths) || error(
        "the chains in $out have different numbers of draws, $lengths; pass `nsamples` to select the same draws of each"
    )
    return Comrade.PosteriorSamples(
        hcat(Comrade.postsamples.(per_chain)...), hcat(Comrade.samplerstats.(per_chain)...);
        metadata = Dict(:sampler => :KineticSamplers, :chains => length(per_chain))
    )
end

function _chain_dirs(out)
    names = filter(n -> occursin(r"^chain_\d+$", n), readdir(out))
    isempty(names) && error("no chains (chain_1, chain_2, …) in $out")
    sort!(names; by = n -> parse(Int, n[7:end]))
    return [joinpath(out, n) for n in names]
end

check_chains(target, rngs, alg::KineticSamplers.MCLMC, params, chains, block, d) =
    check_mclmc(target, rngs, alg, params, chains, block, d)
check_chains(target, rngs, ::KineticSamplers.WALNUTS, params, chains, block, d) = (chains, nothing)

# Runs MCLMC_CHECK_BLOCKS blocks and applies KineticSamplers' end-of-warmup energy check to them,
# logging each chain's statistics first; a failed check is a warning.
function check_mclmc(target, rngs, alg, params, chains, block, d)
    recorder = KineticSamplers.NoPositions()
    trace = nothing
    energies = NamedTuple[]
    for _ in 1:MCLMC_CHECK_BLOCKS
        chains, trace = KineticSamplers.run_block(
            rngs, target, alg, params, chains, trace, block, recorder
        )
        push!(
            energies, (;
                energy_error = Array(trace.energy_error), diverged = Array(trace.diverged),
                step_halvings = Array(trace.step_halvings),
            )
        )
    end
    for k in axes(first(energies).energy_error, 2)
        chain_k = [map(a -> a[:, k:k], e) for e in energies]
        st = KineticSamplers.energy_statistics(chain_k, d)
        @info @sprintf(
            "MCLMC check, chain %d: halved %.3g, divergent %.3g, energy error variance per dimension %.3g",
            k, st.halved, st.divergent, st.energy_var
        )
    end
    try
        KineticSamplers.check_energy(energies, d)
    catch err
        err isa ErrorException || rethrow()
        @warn "MCLMC check failed, sampling continues: $(err.msg)"
    end
    return chains, trace
end

_host_mass(m) = (; stds = Array(m.stds), vecs = Array(m.vecs), vals = Array(m.vals))
_host_params(p::KineticSamplers.WalnutsParameters) =
    (; p.step_size, inverse_mass = _host_mass(p.inverse_mass))
_host_params(p) = (; p.step_size, p.trajectory_length, inverse_mass = _host_mass(p.inverse_mass))

function _log_warmup(::KineticSamplers.MCLMC, params, history, d, seconds)
    c = history.curvature
    @info @sprintf(
        "MCLMC warmup done in %.0f s: step size %.4g, trajectory length %.4g = %.2f √d, inverse mass rank %d%s",
        seconds, params.step_size, params.trajectory_length,
        params.trajectory_length / sqrt(d), size(params.inverse_mass.vecs, 2),
        isnothing(c) ? ", no step-size cap" : @sprintf(
                ", step-size cap %.4g at the %.2g quantile curvature %.4g (median %.4g, max %.4g over %d positions)",
                c.step_size_cap, CAP_QUANTILE, c.curvature, c.median, c.max, c.positions
            )
    )
    return nothing
end

# The worst estimates of a round of `KineticSamplers.measure_bias`, for the run log.
function _bias_text(r, d)
    where_(j) = j <= d ? "coordinate $j" : "direction $(j - d) of the inverse mass"
    return @sprintf(
        "largest mean bias %.3g ± %.2g sd (%s), largest variance bias %.3g ± %.2g (%s); %d means and %d variances unresolved",
        r.worst_mean.bias, r.worst_mean.se, where_(r.worst_mean.monitored),
        r.worst_variance.bias, r.worst_variance.se, where_(r.worst_variance.monitored),
        r.unresolved_means, r.unresolved_variances
    )
end

_bias_tolerances(check) = @sprintf(
    "the tolerances of %.3g sd in a mean or %.3g of a variance (Monte Carlo errors at ESS %d)",
    check.max_mean_bias, check.max_variance_bias, BIAS_ESS
)

"""
    bias_check() -> KineticSamplers.BiasCheck

The bias check of an MCLMC run: coupled runs of $(BIAS_STEPS) steps, doubled up to
$(BIAS_MAX_STEPS) while a round is undecided, with the Monte Carlo errors at an effective sample
size of $(BIAS_ESS) as the tolerances.
"""
bias_check() = KineticSamplers.BiasCheck(; steps = BIAS_STEPS, ess = BIAS_ESS, max_steps = BIAS_MAX_STEPS)

"""
    bias_budget(strategy) -> Int

The most coupled steps per chain the MCLMC step-size tuning runs in all. Each runs three
integrator steps, one with one substep and two with two, so the budget's coupled steps cost
$(BIAS_BUDGET) of the gradients of sampling, `strategy.nsample` times `strategy.thin` times
`strategy.chunk_size` steps per chain.
"""
bias_budget(strategy) =
    max(1, floor(Int, BIAS_BUDGET * strategy.nsample * strategy.thin * strategy.chunk_size / 3))

# Logs the bias of MCLMC at `params` from `KineticSamplers.measure_bias` with `check`, with a
# warning if it exceeds the tolerances, and returns the chains it ran.
function _bias_diagnostic(rngs, target, alg, params, chains, d, check)
    t0 = time()
    chains, r = KineticSamplers.measure_bias(rngs, target, alg, params, chains, check)
    text = @sprintf(
        "MCLMC bias at the sampling step size, from coupled runs of %d steps with 1 and 2 substeps: %s (%.0f s)",
        check.steps, _bias_text(r, d), time() - t0
    )
    if r.rejected
        @warn text * "; above " * _bias_tolerances(check) * ", so the draws are biased at least there"
    else
        @info text
    end
    _flush_logs()
    return chains
end

# `params` with the step size of `KineticSamplers.tune_step_size` with `check`, logging each
# round, and the chains it ran.
function _tune_step_size(rngs, target, alg, params, chains, d, check, max_total_steps)
    t0 = time()
    function report(r)
        @info @sprintf(
            "MCLMC bias at step size %.4g, from coupled runs of %d steps with 1 and 2 substeps: %s; %s (%.0f s)",
            r.step_size, r.steps, _bias_text(r, d),
            r.accepted ? "accepted" : r.rejected ? @sprintf("rejected, %.2g times the tolerance", r.ratio) : "undecided",
            time() - t0
        )
        _flush_logs()
        return nothing
    end
    tuned, chains, rounds = KineticSamplers.tune_step_size(
        rngs, target, alg, params, chains, check; report, max_rounds = BIAS_MAX_ROUNDS,
        max_total_steps
    )
    @info @sprintf(
        "MCLMC step size %.4g from the bias tuning (warmup: %.4g) %s %s, after %d rounds (%.0f s)",
        tuned.step_size, params.step_size,
        last(rounds).accepted ? "within" : "unmeasured; no round was within",
        _bias_tolerances(check), length(rounds), time() - t0
    )
    _flush_logs()
    return tuned, chains
end

function _log_warmup(::KineticSamplers.WALNUTS, params, history, d, seconds)
    @info @sprintf(
        "WALNUTS warmup done in %.0f s: step size %.4g, inverse mass rank %d after %d refits",
        seconds, params.step_size, size(params.inverse_mass.vecs, 2), count(history.refit)
    )
    return nothing
end

"""
    ChainDrawWriter(dir, per_file=$(KINETIC_DRAWS_PER_FILE))

Draws of one chain written to `dir` as a Comrade sample directory (`samples/output_scan_*.jls`
and `parameters.jls`), `per_file` draws per file, readable with `Comrade.load_samples(dir)`
while the chain runs. A `dir` that holds draws is appended to.
"""
mutable struct ChainDrawWriter
    dir::String
    per_file::Int
    samples::Vector{Any}
    stats::Vector{NamedTuple}
    files::Int
    draws::Int
end

function ChainDrawWriter(dir, per_file = KINETIC_DRAWS_PER_FILE)
    p = joinpath(dir, "parameters.jls")
    isfile(p) || return ChainDrawWriter(mkpath(dir), per_file, Any[], NamedTuple[], 0, 0)
    o = deserialize(p).params
    o.stride == per_file && o.nsamples == o.nfiles * o.stride || error(
        "$dir holds $(o.nsamples) draws in $(o.nfiles) files of $(o.stride); draws can be appended only to full files of $per_file"
    )
    return ChainDrawWriter(dir, per_file, Any[], NamedTuple[], o.nfiles, o.nsamples)
end
# The draws per chain already written to `chain_dirs` (0 without any), which must agree.
function _written_draws(chain_dirs)
    counts = map(chain_dirs) do dir
        p = joinpath(dir, "parameters.jls")
        isfile(p) ? deserialize(p).params.nsamples : 0
    end
    allequal(counts) || error(
        "the chains in $(dirname(first(chain_dirs))) hold different numbers of draws, $counts; sampling cannot resume"
    )
    return first(counts)
end

# Appends the scalar parameters (see `scalar_leaves`) of the first `n` draws of each chain in
# `chain_dirs` to `scalars`, reading $(KINETIC_READ_DRAWS) draws at a time.
function _read_scalars!(scalars, names, chain_dirs, n)
    for (k, dir) in pairs(chain_dirs), lo in 1:KINETIC_READ_DRAWS:n
        for θ in Comrade.postsamples(Comrade.load_samples(dir, lo:min(lo + KINETIC_READ_DRAWS - 1, n)))
            leaves = scalar_leaves(θ)
            isempty(names) && append!(names, first.(leaves))
            push!(scalars[k], last.(leaves))
        end
    end
    return scalars
end

function push_draw!(w::ChainDrawWriter, θ; stats...)
    push!(w.samples, θ)
    push!(w.stats, NamedTuple(stats))
    length(w.samples) == w.per_file && flush_draws!(w)
    return w
end

function flush_draws!(w::ChainDrawWriter)
    isempty(w.samples) && return w
    w.files += 1
    w.draws += length(w.samples)
    ps = Comrade.PosteriorSamples(map(identity, w.samples), StructArray(map(identity, w.stats)))
    mkpath(joinpath(w.dir, "samples"))
    serialize(
        joinpath(w.dir, "samples", @sprintf("output_scan_%05d.jls", w.files)),
        (samples = Comrade.postsamples(ps), stats = Comrade.samplerstats(ps))
    )
    serialize(
        joinpath(w.dir, "parameters.jls"),
        (; params = Comrade.DiskOutput(abspath(w.dir), w.files, w.per_file, w.draws))
    )
    empty!(w.samples)
    empty!(w.stats)
    return w
end

"""
    WarmupRecord(dir, K)

The warmup of `K` chains as it runs, under `dir`. `record_block!` keeps each chain's
log density after every block; `write_trace` writes them to `dir/trace.jls` as
`(; block, phase, steps, seconds, logp)`, with `logp` a chains × blocks matrix and `seconds`
counted from the start of the warmup. `record_positions!` writes the chains' positions
as one draw each of the Comrade sample directories `dir/chain_k`, one draw per file, with the
statistics `logp`, `phase`, `block` and `steps`.
"""
struct WarmupRecord
    dir::String
    block::Vector{Int}
    phase::Vector{Symbol}
    steps::Vector{Int}
    seconds::Vector{Float64}
    logp::Vector{Vector{Float64}}
    writers::Vector{ChainDrawWriter}
end

WarmupRecord(dir, K) = WarmupRecord(
    mkpath(dir), Int[], Symbol[], Int[], Float64[], Vector{Float64}[],
    [ChainDrawWriter(joinpath(dir, "chain_$k"), 1) for k in 1:K]
)

function record_block!(r::WarmupRecord, info, seconds, logp)
    length(logp) == length(r.writers) ||
        throw(DimensionMismatch("$(length(logp)) log densities for $(length(r.writers)) chains"))
    push!(r.block, info.block)
    push!(r.phase, info.phase)
    push!(r.steps, info.steps)
    push!(r.seconds, seconds)
    push!(r.logp, collect(Float64, logp))
    return r
end

function record_positions!(r::WarmupRecord, info, logp, positions)
    length(positions) == length(r.writers) == length(logp) || throw(
        DimensionMismatch("$(length(positions)) positions and $(length(logp)) log densities for $(length(r.writers)) chains")
    )
    for (w, x, l) in zip(r.writers, positions, logp)
        push_draw!(w, x; logp = l, info.phase, info.block, info.steps)
    end
    return r
end

# Written to a temporary file and renamed, so `trace.jls` is always a complete trace.
function write_trace(r::WarmupRecord)
    file = joinpath(r.dir, "trace.jls")
    logp = isempty(r.logp) ? zeros(length(r.writers), 0) : reduce(hcat, r.logp)
    serialize(file * ".tmp", (; r.block, r.phase, r.steps, r.seconds, logp))
    mv(file * ".tmp", file; force = true)
    return file
end

"""
    warn_trends(samples; threshold = 1)

Warns about the scalar parameters that trend over `samples`, a vector over time of the
`scalar_leaves` of every chain: those whose mean moves between the first and second half of the
samples in the same direction in every chain, by more than `threshold` times the standard
deviation within a half, pooled over chains and halves. Needs at least four samples and two
chains, and warns that it cannot check otherwise.
"""
function warn_trends(samples; threshold = 1)
    n = length(samples)
    K = isempty(samples) ? 0 : length(first(samples))
    if n < 4 || K < 2
        @warn "the warmup's final phase gave $n positions of $K chains, too few to check " *
            "whether its scalar parameters still trend; set sampler.final to at least " *
            "$(4 * FINAL_TREND_BLOCKS) blocks of steps"
        return String[]
    end
    names = first.(first(first(samples)))
    all(s -> all(c -> first.(c) == names, s), samples) ||
        error("the scalar parameters differ between the positions of the warmup's final phase")
    halves = (1:(n ÷ 2), (n ÷ 2 + 1):n)
    trends = Pair{String, Float64}[]
    for (j, name) in pairs(names)
        v = [last(samples[t][k][j]) for t in 1:n, k in 1:K]
        shift = [mean(v[last(halves), k]) - mean(v[first(halves), k]) for k in 1:K]
        spread = sqrt(mean(var(v[h, k]) for h in halves, k in 1:K))
        spread > 0 && (all(>(0), shift) || all(<(0), shift)) || continue
        z = abs(mean(shift)) / spread
        z > threshold && push!(trends, name => z)
    end
    isempty(trends) && return String[]
    sort!(trends; by = last, rev = true)
    shown = join((@sprintf("%s %.1f", k, z) for (k, z) in first(trends, 10)), ", ")
    @warn "$(length(trends)) scalar parameters still trend at the end of the warmup, in every " *
        "chain the same way (shift between the halves of the final phase in within-half " *
        "standard deviations): $shown$(length(trends) > 10 ? ", …" : ""); the first draws " *
        "may not yet be from the typical set"
    return first.(trends)
end
