# One-call entry point: run the whole pipeline from four TOML config paths. Used by the
# `drivers/main.jl` command and convenient for library use.

"""
    image_from_toml(image, instrument, data, fitting;
                    outpath="Runs/run", restart=false) -> String

Parse the four config TOMLs (image, instrument, data, fitting), build the corresponding
Comrade objects, copy the configs next to the run for provenance, and run
[`comrade_imager`](@ref). Returns the output base path. For MCLMC and WALNUTS, the SHA-256
digests of the image, instrument and data TOMLs and of the data and array files go to the
warmup checkpoints, and a resumed warmup throws if any of these files has changed (see
[`sample_kinetic`](@ref)); the sampler settings of the fitting TOML are checked separately.

The sampler is always NUTS; whether it (and the optimizer) run on Reactant is the single
`run.use_reactant` flag in the fitting TOML — it is not a per-invocation override.
`restart`, by contrast, is a one-off run action
(resume from a previously serialized optimum at `outpath`) and is passed at call time.
"""
function image_from_toml(
        image::AbstractString, instrument::AbstractString, data::AbstractString,
        fitting::AbstractString; outpath::AbstractString = "Runs/run", restart::Bool = false
    )
    @info "Reading configs"
    skycfg = TOML.parsefile(image)
    intcfg = TOML.parsefile(instrument)
    datacfg = TOML.parsefile(data)
    fitcfg = TOML.parsefile(fitting)

    strategy = build_fitting_config(fitcfg)
    intm = build_instrument_config(intcfg)
    # `base_dir` lets `[paths] path_mode = "toml"` resolve relative file/array paths against
    # the data TOML's own directory (so a config dir is portable). `polrep` (from the image
    # config) selects the data product: TotalIntensity fits complex visibilities, polarized
    # models fit coherencies.
    dcoh = build_data_config(
        datacfg; base_dir = dirname(abspath(data)), polrep = sky_polrep(skycfg)
    )
    # Build the sky AFTER the data so the random-field correlation length and the mean-Gaussian
    # width are set from the observation beam (`beamsize(dcoh)`) rather than hand-tuned.
    (skym, imgdata) = build_sky_config(skycfg; beam = Comrade.beamsize(dcoh))

    outdir = dirname(outpath)
    mkpath(isempty(outdir) ? "." : outdir)
    # Copy the configs next to the run for provenance.
    for (f, suffix) in (
            (image, "image.toml"), (instrument, "instrument.toml"),
            (data, "data.toml"), (fitting, "fitting.toml"),
        )
        try
            cp(f, outpath * "_" * suffix; force = true)
        catch err
            @warn "Could not copy config $f" err
        end
    end

    # The fitting config is left out: the checkpoint's own settings check covers the sampler
    # settings it holds, and its other keys (such as `checkpoint_minutes`) may change between
    # a stop and a resume.
    inputs = if strategy.algorithm in KINETIC_ALGORITHMS
        (; file, array) = data_paths(datacfg; base_dir = dirname(abspath(data)))
        input_digests((; image, instrument, data, data_file = file, array_file = array))
    else
        nothing
    end
    return comrade_imager(outpath, skym, intm, dcoh; strategy, imgdata, restart, inputs)
end
