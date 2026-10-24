export read_w90, read_w90_with_chk, write_w90, write_w90_centres

"""
    $(SIGNATURES)

Read `win` and `mmn` files, and read `amn`/`eig` files if they exist.

# Keyword arguments
- ortho_amn: Lowdin orthonormalization after reading `amn` matrices.
    Should be `true` for most cases, since usually the input `amn` matrices are
    not guaranteed to be unitary or semi-unitary.
- use_mmn_bvecs: use the b-vectors in `mmn` file instead of regenerating them.
- use_bvec_file: read b-vectors and their finite-difference weights directly
    from `\$prefix.bvec` (written by wannier90 with `write_bvec = .true.`).
    Required when wannier90 was run with higher-order finite differences
    (`higher_order_n > 1` or `higher_order_nearest_shells`), where the parallel
    b-vectors at multiple norms break [`compute_bweights`](@ref). Reuses
    `kpb_k`/`kpb_G` from the `mmn` file, so it is incompatible with
    `use_mmn_bvecs=false`.
- kstencil_algo: algorithm to generate `KspaceStencil` if `use_mmn_bvecs` is `false`.
    Default is `generate_kspace_stencil`.
"""
function read_w90(prefix::AbstractString; ortho_amn::Bool=true, use_mmn_bvecs::Bool=true, use_bvec_file::Bool=false, kstencil_algo::KspaceStencilAlgorithm=default_kstencil_algo())
    win = read_win(prefix * ".win")
    nbands = win.num_bands
    nwann = win.num_wann
    lattice = win.unit_cell_cart
    recip_lattice = reciprocal_lattice(lattice)
    nkpts = length(win.kpoints)

    isfile(prefix * ".mmn") || error("$(prefix).mmn file does not exist")
    overlaps, kpb_k, kpb_G = read_mmn(prefix * ".mmn")
    nbvecs = length(overlaps[1])
    # check consistency for mmn
    @assert nkpts == length(overlaps) "different n_kpoints in mmn and win files"
    @assert (nbands, nbands) == size(overlaps[1][1]) "different n_bands in mmn and win files"

    if use_bvec_file
        use_mmn_bvecs || error(
            "use_bvec_file=true is incompatible with use_mmn_bvecs=false: " *
            "the bvec path reuses kpb_k/kpb_G from the mmn file"
        )
        isfile(prefix * ".bvec") || error(
            "$(prefix).bvec not found; rerun wannier90 with write_bvec = .true. " *
            "or set use_bvec_file=false"
        )
        atol = get(win, :kmesh_tol, default_w90_kmesh_tol())
        kstencil = read_bvec_kstencil(
            prefix, recip_lattice, win.mp_grid, win.kpoints, kpb_k, kpb_G; atol
        )
        @assert n_bvectors(kstencil) == nbvecs "different n_bvectors in bvec and mmn files"
    elseif use_mmn_bvecs
        kstencil = KspaceStencil(recip_lattice, win.kpoints, kpb_k, kpb_G)
    else
        atol = get(win, :kmesh_tol, default_w90_kmesh_tol())
        kstencil = generate_kspace_stencil(
            recip_lattice, win.mp_grid, win.kpoints, kstencil_algo; atol
        )
        @assert n_bvectors(kstencil) == nbvecs "different n_bvectors in mmn and win files"
        @assert kstencil.kpb_k == kpb_k "auto generated kpb_k are different from mmn file"
        @assert kstencil.kpb_G == kpb_G "auto generated kpb_G are different from mmn file"
        # overlaps = zeros_overlap(ComplexF64, nkpts, nbvecs, nbands)
        # @warn "$prefix.mmn file does not exist, set M to zeros"
    end

    if isfile(prefix * ".amn")
        if ortho_amn
            gauges = read_amn_ortho("$prefix.amn")
        else
            gauges = read_amn("$prefix.amn")
        end
        @assert nkpts == length(gauges) "different n_kpoints in amn and win files"
        @assert (nbands, nwann) == size(gauges[1]) "different n_bands or n_wannier in amn and win files"
    else
        gauges = zeros_gauge(ComplexF64, nkpts, nbands, nwann)
        @warn "$prefix.amn file does not exist, set U to zeros"
    end

    disentangle = nbands != nwann
    if isfile(prefix * ".eig")
        eigenvalues = read_eig(prefix * ".eig")
        @assert nkpts == length(eigenvalues) "different n_kpoints in eig and win files"
        @assert nbands == length(eigenvalues[1]) "different n_bands in eig and win files"
    else
        @assert !disentangle "eig file is required for disentanglement"
        eigenvalues = [zeros(Float64, nbands) for _ in 1:nkpts]
        @warn "$prefix.eig file does not exist, set eigenvalues to zeros"
    end

    if disentangle
        dis_froz_max = get(win, :dis_froz_max, nothing)
        dis_froz_min = get(win, :dis_froz_min, -Inf)
        if isnothing(dis_froz_max)
            frozen_bands = [falses(nbands) for _ in 1:nkpts]
        else
            frozen_bands = get_frozen_bands(eigenvalues, dis_froz_max, dis_froz_min)
        end
    else
        frozen_bands = [falses(nbands) for _ in 1:nkpts]
    end

    atom_labels = map(x -> string(x.first), win.atoms_frac)
    atom_positions = map(x -> x.second, win.atoms_frac)

    return Model(
        lattice,
        atom_positions,
        atom_labels,
        kstencil,
        overlaps,
        gauges,
        eigenvalues,
        frozen_bands,
    )
end

"""
    $(SIGNATURES)

Return a `Model` with U matrix filled by that from a chk file.

# Arguments
- chk: path of chk file to get the unitary matrices.
"""
function read_w90_with_chk(prefix::AbstractString, chk::AbstractString="$prefix.chk")
    model = read_w90(prefix)
    fchk = read_chk(chk)
    model.gauges .= get_U(fchk)
    return model
end

"""
    $(SIGNATURES)

Write `Model` into `eig`, `mmn`, `amn` files.

# Keyword arguments
- `binary`: write `eig`, `mmn`, and `amn` in Fortran binary format
"""
function write_w90(prefix::AbstractString, model::Model; binary::Bool=false)
    outname(suffix) = "$prefix.$suffix"

    write_eig(outname("eig"), model.eigenvalues; binary)
    write_mmn(outname("mmn"), model.overlaps, model.kstencil; binary)
    write_amn(outname("amn"), model.gauges; binary)
    return nothing
end

"""
    write_w90_centres(filename, model::Model, Ω::Spread; translate_home_cell=false)

Write Wannier90's `seedname_centres.xyz` file from a [`Model`](@ref) and a
[`Spread`](@ref) (typically the return value of [`omega`](@ref)).

Atom positions are pulled from `model` (converted from fractional to Cartesian
Å) and WF centers from `Ω.r`.
"""
function write_w90_centres(
    filename::AbstractString, model::Model, Ω::Spread; translate_home_cell::Bool=false
)
    atom_cart = [model.lattice * p for p in model.atom_positions]
    return WannierIO.write_w90_centres(
        filename, Ω.r;
        atom_positions=atom_cart,
        atom_labels=model.atom_labels,
        lattice=model.lattice,
        translate_home_cell,
    )
end
