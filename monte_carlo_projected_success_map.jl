using Distributed
# No addprocs() – control processes with `-p` or via scheduler

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
	x_star = G_star * η_star

	θ_star = [u_, u_]
end

@everywhere function run_row_projected(snr, x_star, θ_star, k_0, T, grid,
	ϵ_values;
	trials = 10, solver_iters = 100, tol = 1e-3)
	[
		monte_carlo_projected(x_star, θ_star, snr, ϵ, k_0, T, grid;
			trials = trials, solver_iters = solver_iters, tol = tol)
		for ϵ in ϵ_values
	]
end

# Sweep parameters
ϵ_samples = 30
snr_samples = 30
trials = 50
solver_iters = 100

ϵ_range_unprojected = logrange(1e-6, 1e0, ϵ_samples)
snr_range = range(5, 90, snr_samples)

# Parallel over SNR values
results_rows = pmap(
	snr -> run_row_projected(snr, x_star, θ_star, k_0, T, grid,
		ϵ_range_unprojected;
		trials = trials, solver_iters = solver_iters),
	snr_range)

# Stack into matrix
projected_success_map = reduce(vcat, [reshape(r, 1, :) for r in results_rows])

# Save
writedlm("results/tsp_monte_carlo_projected_success_map.txt",
	(ϵ_range_unprojected, snr_range, projected_success_map))
