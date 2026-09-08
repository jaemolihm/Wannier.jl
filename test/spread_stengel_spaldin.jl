@testitem "StengelSpaldinSpread ordering" begin
    using Wannier.Datasets
    model = load_dataset("Si2_valence")
    p = StengelSpaldinSpread(model.kstencil)

    nbvecs = Wannier.n_bvectors(model.kstencil)
    # `nnord` must be a permutation of the b-vector slots at every kpoint
    @test all(sort(o) == collect(1:nbvecs) for o in p.nnord)
    # `ibrev` pairs each b with -b, so it is an involution without fixed points
    @test p.ibrev[p.ibrev] == collect(1:nbvecs)
    @test all(p.ibrev .!= 1:nbvecs)
    @test all(
        isapprox(p.bvectors[p.ibrev[ib]], -p.bvectors[ib]; atol = 1.0e-6)
            for ib in 1:nbvecs
    )

    # slot `ib` must hold the same b-vector at every kpoint
    for ik in 1:Wannier.n_kpoints(model.kstencil)
        bvecs = Wannier.get_bvectors(model.kstencil, ik)
        @test all(
            isapprox(bvecs[p.nnord[ik][ib]], p.bvectors[ib]; atol = 1.0e-6)
                for ib in 1:nbvecs
        )
    end
end

@testitem "StengelSpaldinSpread spread" begin
    using Wannier.Datasets
    model = load_dataset("Si2_valence")
    p = StengelSpaldinSpread(model.kstencil)

    Ωmv = omega(model.kstencil, model.overlaps, model.gauges)
    Ωss = omega(p, model.kstencil, model.overlaps, model.gauges)

    # SS changes only the diagonal part
    @test Ωss.ΩI ≈ Ωmv.ΩI
    @test Ωss.ΩOD ≈ Ωmv.ΩOD
    @test !isapprox(Ωss.ΩD, Ωmv.ΩD; atol = 1.0e-8)

    # ΩD is a variance over kpoints, hence non-negative
    @test Ωss.ΩD >= 0
    # Ω = Σ_nb w_b (1 - |S_nb|²) collapses the three parts
    @test Ωss.Ω ≈ sum(Ωss.ω)
    @test Ωss.Ω ≈ Ωss.ΩI + Ωss.ΩOD + Ωss.ΩD

    # the k-averaged overlaps obey S[n, -b] = conj(S[n, b]), which the gradient
    # derivation relies on
    cache = Wannier.Cache(model.kstencil, model.overlaps, model.gauges)
    Wannier.compute_MU_UtMU!(cache, model.kstencil, model.overlaps, model.gauges)
    S = Wannier.compute_ss_overlaps(p, cache.UtMU, Wannier.n_wannier(model))
    @test S[:, p.ibrev] ≈ conj(S)
end

@testitem "StengelSpaldinSpread gradient" begin
    using NLSolversBase
    using Wannier.Datasets
    model = read_w90_with_chk(dataset"Si2_coarse/Si2", dataset"Si2_coarse/outputs/Si2.chk")
    p = StengelSpaldinSpread(model.kstencil)
    fg! = Wannier.get_fg!_maxloc(p, model)

    nb = n_bands(model)
    nw = n_wannier(model)
    nk = n_kpoints(model)
    U = [model.gauges[ik][ib, ic] for ib in 1:nb, ic in 1:nw, ik in 1:nk]
    G = zero(U)
    fg!(nothing, G, U)

    # Use finite difference as reference
    Uinit = deepcopy(U)
    d = NLSolversBase.OnceDifferentiable(
        x -> fg!(1.0, nothing, x), Uinit, real(zero(eltype(Uinit)))
    )
    G_ref = NLSolversBase.gradient!(d, U)

    @test isapprox(G, G_ref; atol = 1.0e-7)
end

@testitem "StengelSpaldinSpread max_localize" begin
    using LinearAlgebra
    using Wannier.Datasets
    model = load_dataset("Si2_valence")
    p = StengelSpaldinSpread(model.kstencil)

    Ωi = omega(p, model.kstencil, model.overlaps, model.gauges)
    Umin = max_localize(p, model; max_iter = 30)
    Ωf = omega(p, model.kstencil, model.overlaps, Umin)

    @test Ωf.Ω < Ωi.Ω
    # the gauge stays on the unitary manifold
    @test all(isapprox(U' * U, I; atol = 1.0e-10) for U in Umin)
    # ΩI is gauge invariant, so minimising must not move it
    @test Ωf.ΩI ≈ Ωi.ΩI
end
