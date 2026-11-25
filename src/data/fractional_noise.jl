"""
    add_coherency_noise(data, ferr, ferr_leak = 0.0)

A copy of `data` with fractional noise added in quadrature to each measurement.

For coherency matrices, each element Vᵢⱼ (station a's feed i, station b's feed j) gets

    σᵢⱼ² → σᵢⱼ² + ferr² |Vᵢⱼ|² + ferr_leak² (|V_{īj}|² + |V_{ij̄}|²) / 2,

where ī and j̄ are the other feed of each station. The `ferr` term is the error from fractional
gain errors on the feeds (residual gains, bandpass, delays). The `ferr_leak` term is the error from
leakage errors on the feeds, which move the other feed's signal into Vᵢⱼ: station a's leakage
brings in V_{īj} and station b's brings in V_{ij̄}. Both fractions are per baseline, so on a
circular-feed baseline `ferr = ferr_leak = f` adds f |I| to every element, up to terms in V.
Both terms hold in the feed frame of the data (no field-rotation correction applied) for any feed
basis, and for circular feeds in any frame. Entries missing from the data (NaN, e.g. the
absent feed of a single-feed station) contribute nothing and keep their noise.

Other data products have no feeds; they get Comrade's `add_fractional_noise(data, ferr)`, and
`ferr_leak` must be zero.
"""
function add_coherency_noise(data, ferr, ferr_leak = 0.0)
    iszero(ferr_leak) || error("ferr_leak = $ferr_leak applies only to coherency data, not $(typeof(data))")
    return add_fractional_noise(data, ferr)
end

function add_coherency_noise(dvis::Comrade.EHTObservationTable{<:Comrade.EHTCoherencyDatum}, ferr, ferr_leak = 0.0)
    dvis = deepcopy(dvis)
    map!(dvis[:noise], dvis[:noise], dvis[:measurement]) do e, m
        L = leakage_power(m)
        return @. ifelse(isfinite(m), sqrt(e^2 + ferr^2 * abs2(m) + ferr_leak^2 * L / 2), e)
    end
    return dvis
end

# |V_{īj}|² + |V_{ij̄}|² for each element of the 2×2 coherency matrix `m`: the parallel hands
# receive leakage from the cross hands and the reverse. Missing (NaN) entries count as zero.
function leakage_power(m)
    Base.require_one_based_indexing(m)
    size(m) == (2, 2) || throw(DimensionMismatch("coherency matrix must be 2×2, got size $(size(m))"))
    a2(x) = isfinite(x) ? abs2(x) : zero(abs2(x))
    p = a2(m[1, 1]) + a2(m[2, 2])
    c = a2(m[1, 2]) + a2(m[2, 1])
    return SMatrix{2, 2}(c, p, p, c)
end
