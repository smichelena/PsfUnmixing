using Distributed
addprocs()  # or let your cluster scheduler handle it via mpiexec

@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using PsfUnmixing, LinearAlgebra, DelimitedFiles

# Parameters
@everywhere begin
    p_ = 25
    u_ = 1e-1
    Δ_ = 4e-1
    L = 3
    N = 10001
    grid = range(-1.0, 1.0, N)

    umin = 1e-5
    umax = 2e-1

    k_0 = (u, t) -> laplace_0_n(s_laplace(u, p_, umin), t, p_, umin = umin)
    k_1 = (u, t) -> laplace_1_n(s_laplace(u, p_, umin), t, p_, umin = umin)

    T = generate_spike_groups((-1.0, 1.0), 2, L, Δ_);

    G_star = multi_block(k_0, [u_, u_], T, grid);
    η_ = rand(6);
    x_ = G_star * η_;
    η_star = η_ ./ norm(x_);
    u_star = vcat([u_, u_], η_star)
    x_star = G_star * η_star;

    snr_ = 24.0

    ϵ_range = logrange(1e-10, 1e0, 50)
end

# Parallel execution
@everywhere function run_trial(ϵ_perb, x_star, u_star, snr, T, grid, k_0, k_1)
    monte_carlo_unprojected(
        x_star, u_star, snr, T, grid, k_0, k_1;
        ϵ = ϵ_perb,
        trials = 50,
        solver_iters = 100,
        tol = 1e-3,
    )
end

success_rate = pmap(ϵ -> run_trial(ϵ, x_star, u_star, snr_, T, grid, k_0, k_1), ϵ_range)

writedlm("results/tsp_monte_carlo_unprojected_snr_24.txt", success_rate)