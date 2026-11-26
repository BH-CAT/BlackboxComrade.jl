# Data loaders for the two supported file formats. Both apply scan/fixed-interval averaging,
# optional time-range selection, and fractional-noise inflation. The Comrade data product is
# chosen from the sky model's polarization representation (`polrep`): a Stokes-I-only sky
# (`TotalIntensity`) fits complex visibilities, while a polarized sky fits coherency matrices.

# Map a polarization representation to the Comrade data-product type to extract.
data_product(::TotalIntensity) = Visibilities
data_product(::PolRep) = Coherencies

function build_data_uvfits(
        file::String, array::Union{String, Nothing} = nothing;
        avg = "scan",
        scan_gap::Real = 20.0,
        ferr::Float64 = 0.005,
        ferr_leak::Float64 = 0.0,
        trange = nothing,
        IF = nothing,
        polrep::PolRep = PolExp(),
        conjugate::Bool = false,
        ignore_feed_labels::Bool = false,
    )

    uvd = VLBIFiles.load(VLBIFiles.UVData, file)
    ignore_feed_labels && force_circular_feed_labels!(uvd)

    # `avg = "scan"` averages each scan coherently; a time gap longer than `scan_gap` seconds
    # between consecutive records starts a new scan.
    if avg == "scan"
        scan_gap > 0 || error("scan_gap must be positive (seconds); got $scan_gap")
        tavg = VLBI.GapBasedScans(; min_gap = Float64(scan_gap) * VLBIFiles.Unitful.u"s")
    else
        tavg = VLBI.FixedTimeIntervals(parse(Float64, avg) * VLBIFiles.Unitful.u"s")
    end

    # The array file describes the antenna feed-rotation parameters; when given we apply it
    # via `reset_mounts!` after extracting the data table. It is optional: without it the
    # feed-rotation/mount metadata already in the data file is kept (fine for Stokes I; for
    # polarized fits supply an array file if the data-file mounts are wrong).
    #
    # `IF` selects a single intermediate frequency (1-based index into the sorted unique
    # frequencies). When set we keep the IFs separate (`frequency_average = false`) and filter
    # to the requested one; otherwise the default frequency-averaged band is extracted.
    product = data_product(polrep)
    dvis = extract_table(
        uvd, product(;
            time_average = tavg,
            frequency_average = isnothing(IF),
        )
    )
    # Comrade builds the array's scan table with VLBIData's default gap (1 minute) whatever
    # `time_average` is. Rebuild it with the averaging strategy, so records averaged into
    # separate scans also get separate scan-segmented instrument parameters (see
    # ptiede/Comrade.jl#480).
    avg == "scan" && (dvis = set_scans(dvis, VLBI.scan_intervals(tavg, VLBI.uvtable(uvd))))
    if !isnothing(IF)
        ifs = sort(unique(dvis.config.datatable.Fr))
        (IF isa Integer && 1 <= IF <= length(ifs)) ||
            error("IF=$IF is invalid; IF is 1-based and the data has $(length(ifs)) IF(s) (use 1..$(length(ifs))).")
        dvis = filter(d -> d.baseline.Fr == ifs[IF], dvis)
        @info "Selected IF $IF/$(length(ifs)) (frequency = $(ifs[IF] / 1.0e9) GHz)"
    end
    if !isnothing(trange)
        dvis = filter(d -> d.baseline.Ti ∈ trange, dvis)
    end
    if isnothing(array)
        @info "No array file specified; keeping the feed-rotation/mount metadata from the data file as-is."
    else
        reset_mounts!(dvis, array)
    end

    dvis = add_coherency_noise(dvis, ferr, ferr_leak)
    conjugate && (dvis = conjugate_data(dvis))
    return dvis
end

"""
    force_circular_feed_labels!(uvd::VLBIFiles.UVData)

Relabel every antenna's feeds as `(:R, :L)` so the file's `RR/LL/RL/LR` slots are read as-is
for every baseline. Use this when the antenna table (POLTYA/POLTYB) marks some stations
linear: VLBIFiles then translates their products to mixed labels (e.g. `YR`) that the Comrade
extractor rejects, and the labels may not even be right. After loading, declare the linear
stations with `[flags] corr_polbasis` (e.g. `{site = "AA", R = "X", L = "Y"}`), which is the
single place the feed bookkeeping is meant to live. Only valid for circular-slot files.
"""
function force_circular_feed_labels!(uvd)
    stokes = uvd.header.stokes
    all(s -> s in (:RR, :LL, :RL, :LR), stokes) || error(
        "ignore_feed_labels requires a file whose stokes axis is RR/LL/RL/LR; got $stokes"
    )
    for arr in uvd.ant_arrays, (id, ant) in pairs(arr.antennas)
        ant.poltypes == (:R, :L) && continue
        @info "ignore_feed_labels: $(ant.name) antenna-table feeds $(ant.poltypes) treated as (R, L)"
        arr.antennas[id] = VLBIFiles.VLBIData.Antenna(;
            name = ant.name, xyz = ant.xyz, mount_type = ant.mount_type,
            poltypes = (:R, :L), feed_offsets = ant.feed_offsets,
        )
    end
    return uvd
end

"""
    conjugate_data(dvis) -> dvis

Flip the Fourier-phase sign convention of every datum: uvfits written with the opposite
convention (e.g. FITS-IDI-style phases vs. the AIPS/ehtim convention Comrade assumes) store
`V(-u,-v)` under the `(u,v)` label, i.e. the Hermitian adjoint of the coherency matrix (the
complex conjugate for a Stokes I visibility). Also transposes the noise matrix so the
RL/LR entries follow their measurements. Closure phases change sign under this operation.
"""
function conjugate_data(dvis::Comrade.EHTObservationTable{T}) where {T}
    meas = map(adjoint, dvis.measurement)
    noise = map(transpose, dvis.noise)
    return Comrade.EHTObservationTable{T}(meas, noise, dvis.config)
end

"""
    set_scans(dvis, intervals) -> dvis

Replace the scan table of `dvis` with `intervals` (DateTime intervals, as returned by
`VLBIData.scan_intervals`), converted to hours after midnight of the first scan's day, the
convention of the data's `Ti`. Throws if an averaged time falls outside every new scan.
"""
function set_scans(dvis, intervals)
    start, stop = map(minimum, intervals), map(maximum, intervals)
    issorted(start) || error("scan intervals are not sorted by start time")
    midnight = Dates.DateTime(Dates.Date(first(start)))
    hours(dt) = Dates.value(dt - midnight) / 3.6e6
    scans = StructArray((; start = map(hours, start), stop = map(hours, stop)))
    for t in unique(arrayconfig(dvis)[:Ti])
        any(i -> scans.start[i] <= t <= scans.stop[i], eachindex(scans)) ||
            error("averaged time $t h lies outside every scan of the rebuilt scan table")
    end
    arr = arrayconfig(dvis)
    newarr = @set arr.scans = scans
    return @set dvis.config = newarr
end

"""
    repair_scan_coverage(dvis) -> dvis

Widen the scan table so every timestamp lies in exactly one scan, as scan-segmented
instrument parameters require. See [`cover_times`](@ref).
"""
function repair_scan_coverage(dvis)
    arr = arrayconfig(dvis)
    start, stop, orphans = cover_times(arr.scans.start, arr.scans.stop, unique(arr[:Ti]))
    isempty(orphans) && return dvis
    @warn "$(length(orphans)) averaged timestamp(s) fell outside the scan table; " *
        "stretched the nearest scan edges to cover them." orphans
    newarr = @set arr.scans = StructArray((; start, stop))
    return @set dvis.config = newarr
end

scanseg_intervals(start, stop) =
    Comrade.timestamps(ScanSeg(), (; scans = StructArray((; start, stop)), mjd = 0))

scans_containing(t, intervals) = count(s -> t ∈ s, intervals)

"""
    cover_times(start, stop, times; pad = 1.0e-3 / 3600) -> (start, stop, orphans)

Return copies of the scan edges `start` and `stop` (hours) in which each of `times` lies in
exactly one Comrade `ScanSeg` interval, and the `orphans` that no scan contained. Each
orphan's nearest scan is widened to `pad` beyond it on both sides; an edge exactly at the
time can round to the wrong side. Throws if a time still lies in zero or several scans.
"""
function cover_times(start, stop, times; pad = 1.0e-3 / 3600)
    start, stop = copy(start), copy(stop)
    orphans = eltype(times)[]
    for t in times
        scans_containing(t, scanseg_intervals(start, stop)) > 0 && continue
        push!(orphans, t)
        i = argmin(i -> max(start[i] - t, t - stop[i]), eachindex(start, stop))
        start[i] = min(start[i], t - pad)
        stop[i] = max(stop[i], t + pad)
    end
    intervals = scanseg_intervals(start, stop)
    for t in times
        n = scans_containing(t, intervals)
        n == 1 || error("averaged time $t h lies in $n scans of the repaired scan table")
    end
    return start, stop, orphans
end

function build_data_dlist(
        file::String, array::Union{String, Nothing} = nothing;
        avg = "scan",
        ferr::Float64 = 0.005,
        ferr_leak::Float64 = 0.0,
        trange = nothing,
        polrep::PolRep = PolExp(),
    )

    # Unlike uvfits, the array file is required for dlist: it provides the antenna table used
    # to construct the array configuration (the dlist itself carries no array metadata).
    isnothing(array) && error(
        "dlist files require an 'array' file (it provides the antenna table used to build " *
            "the array configuration)."
    )

    # dlist files are mixed-polarization coherency matrices by construction; there is no
    # Stokes-I-visibility extraction path for them.
    polrep isa TotalIntensity && error(
        "dlist files are polarized coherencies; TotalIntensity (complex-visibility) fitting " *
            "is not supported for dlist. Use a polarized polrep (PolExp/Poincare) or convert " *
            "the data to uvfits."
    )

    avg != "scan" && @warn "Only 'scan' averaging is supported for dlist files. Ignoring the --avg flag."

    dcoh0 = read_dlist(file, array)
    if !isnothing(trange)
        dcoh1 = filter(x -> trange[1] < x.baseline.Ti < trange[2], dcoh0)
    else
        dcoh1 = dcoh0
    end
    dcoh2 = add_coherency_noise(dcoh1, ferr, ferr_leak)
    return dcoh2
end
