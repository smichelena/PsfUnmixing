using Distributed
# Processes still controlled externally (`-p`)

@everywhere begin
    using Pkg
    Pkg.activate(".")
    using PsfUnmixing, LinearAlgebra, Random, DelimitedFiles, ForwardDiff, Statistics

    T = 1.0
    Δ = 0.2
    d = 2
    p = 2 # code only works for p = 2 thus far 

    θ_min = 1e-2
    θ_max = 1e-1

    u = 2

    s_min = s_laplace(θ_min, u, θ_min)
    s_max = s_laplace(θ_max, u, θ_min)

    s_star = 0.5 * (s_max + s_min) * ones(p)

    N = 10001
    grid = range(-1.0, 1.0, N)

    k_0(s, t) = laplace_0_n(s, t, u, umin = θ_min)
    k_1(s, t) = laplace_1_n(s, t, u, umin = θ_min)
    k_2(s, t) = laplace_2_n(s, t, u, umin = θ_min)

    η_star = ones(p*d)

    lines = generate_spike_groups((-T, T), p, d, Δ)

    G_0(s) = multi_block(k_0, s, lines, grid)
    G_1(s) = multi_block(k_1, s, lines, grid)
    G_2(s) = multi_block(k_2, s, lines, grid)

    x_star = G_0(s_star) * η_star
end

@everywhere function block_diag_matrices(matrices::Vector{<:AbstractMatrix})
    total_rows = sum(size.(matrices, 1))
    total_cols = sum(size.(matrices, 2))
    result = zeros(eltype(first(matrices)), total_rows, total_cols)

    row_start = 1
    col_start = 1
    for mat in matrices
        rows, cols = size(mat)
        result[row_start:row_start+rows-1, col_start:col_start+cols-1] .= mat
        row_start += rows
        col_start += cols
    end

    return result
end

@everywhere H_p_analytical(x_obs, θ) = begin
    G0 = G_0(θ)
    G1 = G_1(θ)
    G2 = G_2(θ)
    eta = G0 \ x_obs
    r = x_obs - G0 * eta
    diag_eta = block_diag(collect(Iterators.partition(eta, d)))
    ∇_θr = G1 * diag_eta
    ∇_ηr = G0

    r∇_θr  = - transpose(kron(I(p), r)) * block_diag_matrices([G2[:,1:d], G2[:,(d+1):end]]) * diag_eta
    r∇_ηθr = - transpose(block_diag_matrices([G1[:,1:d], G1[:,(d+1):end]])) * kron(I(p), r)

    ∇_θL  = transpose(∇_θr) * ∇_θr + r∇_θr
    ∇_ηθL = transpose(∇_ηr) * ∇_θr + r∇_ηθr 
    return ∇_θL - transpose(∇_ηθL) * inv(transpose(G0)*G0) * ∇_ηθL
end

@everywhere function run_trial(ϵ_perb, x_star, θ_star, snr_;
    trials = 1000,
)
    successes = 0

    for _ in 1:trials
        # perturb initialization
        θ_init = random_perturbation_projected(θ_star, ϵ_perb)

        # loss with noisy data (but here just x_star at fixed snr)
        σ = sqrt(mean(abs2, x_star) / (10.0^(snr_ / 10.0)))
        x_obs = x_star .+ σ * randn(length(x_star))

        # Hessian at init
        H = H_p_analytical(x_obs, θ_init)

        λmin = minimum(eigvals(Symmetric(H)))
        if λmin > 0
            successes += 1
        end
    end

    return successes / trials
end

ϵ_range = logrange(1e-5, 1e0, 100)
snr = 10.0
trials = 1000

# Parallel execution
success_rate_p = pmap(
    ϵ -> run_trial(
        ϵ,
        x_star,
        s_star,
        snr,
        trials = trials,
    ),
    ϵ_range)

# Save results
writedlm("results/tsp_monte_carlo_convexity_projected_snr_$(snr)_u_$(u).txt", success_rate_p)
