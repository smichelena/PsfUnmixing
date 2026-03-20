using Distributed
# No addprocs() here – use `-p` when starting Julia

@everywhere begin
	using Pkg
	Pkg.activate(".")
	using PsfUnmixing, LinearAlgebra, Random, DelimitedFiles

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

	T = generate_spike_groups((-1.0, 1.0), 2, L, Δ_)

	G_star = multi_block(k_0, [u_, u_], T, grid)
	η_ = rand(6)
	x_ = G_star * η_
	η_star = η_ ./ norm(x_)
	u_star = vcat([u_, u_], η_star)
	x_star = G_star * η_star
end

@everywhere function run_point(snr, x_star, u_star, T, grid, k_0, k_1, ϵ_values;
	trials = 1, solver_iters = 1, tol = 1e-3)
	[
		monte_carlo_unprojected(x_star, u_star, snr, T, grid, k_0, k_1;
			ϵ = ϵv, trials = trials,
			solver_iters = solver_iters, tol = tol)
		for ϵv in ϵ_values
	]
end

# Parameters for sweep
ϵ_samples = 30
snr_samples = 30
trials = 50
solver_iters = 100
ϵ_range_unprojected = logrange(1e-10, 1e0, ϵ_samples)
snr_range = range(25, 30, snr_samples)

# Parallel map over SNR values
results_rows = pmap(
	snr -> run_point(
		snr,
		x_star,
		u_star,
		T,
		grid,
		k_0,
		k_1,
		ϵ_range_unprojected,
        trials = trials,
        solver_iters = solver_iters
	),
	snr_range)

# Stack results into a matrix
unprojected_success_map = reduce(vcat, [reshape(r, 1, :) for r in results_rows])

# Save
writedlm("results/tsp_monte_carlo_unprojected_success_map.txt",
	(ϵ_range_unprojected, snr_range, unprojected_success_map))
