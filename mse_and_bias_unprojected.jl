@everywhere begin
	using Pkg
	Pkg.activate(".")
	using PsfUnmixing,
		LinearAlgebra, ForwardDiff, Random, DelimitedFiles, Statistics

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

	u_star = vcat([u_, u_], η_star)

	G_1 = multi_block(k_1, [u_, u_], T, grid)

	# J_θ = G_1 * block_diag([η_star[1:3], η_star[4:end]])

	# J = hcat(J_θ, G_star) # analitical jacobian hell yeah
	# trace_inv_J = sum(eigvals(Hermitian(transpose(J) * J)) .^ (-1))

	function residual_unprojected(u, x, T, k_0, grid)
		G = multi_block(k_0, u[1:2], T, grid)
		return G * u[3:end] - x
	end

	residual = u -> residual_unprojected(u, x_star, T, k_0, grid)
	J = ForwardDiff.jacobian(residual, u_star)
    trace_inv_J = sum(eigvals(Hermitian(transpose(J) * J)) .^ (-1))
    
	# println(
	# 	"||J_analytic - J_autodiff||/||J_autodiff|| = $(norm(J - J_autodiff)/norm(J_autodiff))",
	# )
end

@everywhere function MSE_bias_unprojected(
	snr,
	x_star,
	u_star,
	k_0,
	k_1,
	T,
	grid;
	ϵ_init = 1e-5,
	trials = 50,
	solver_iters = 100,
)
	u_hats = zeros(length(u_star), trials)

	for i in 1:trials
		x_noisy = add_noise_snr(x_star, snr)
		u_init = random_perturbation(u_star, 2, ϵ_init)

		u_hat = run_solver_unprojected(
			x_noisy,
			T,
			grid,
			k_0,
			k_1,
			u_init,
			solver_iters = solver_iters,
		)

		u_hats[:, i] .= u_hat
	end

	mean_est = mean(u_hats, dims = 2)[:]  # average across trials
	bias_vec = mean_est .- u_star
	bias_norm2 = sum(bias_vec .^ 2)
	mse = mean(sum((u_hats .- u_star) .^ 2, dims = 1) ./ length(u_star))

	return mse, bias_vec, bias_norm2
end

# Sweep parameters
ϵ_init = 1e-3
snr_samples = 20
trials = 50
solver_iters = 100
snr_range = range(15, 90, snr_samples)

crb =
	snr_range .|>
	snr -> (1 / (length(x_star) * 10.0^(snr / 10.0))) * trace_inv_J

# Parallel run — returns tuple for each SNR
results = pmap(
	snr -> MSE_bias_unprojected(
		snr,
		x_star,
		u_star,
		k_0,
		k_1,
		T,
		grid,
		ϵ_init = ϵ_init,
		trials = trials,
		solver_iters = solver_iters,
	),
	snr_range,
)

mse_vals    = [r[1] for r in results]
bias_vecs   = [r[2] for r in results]  # vector for each SNR
bias_norm2s = [r[3] for r in results]

# Save all in one file: SNR, CRB, MSE, bias_norm², bias1, bias2
data = [snr_range crb mse_vals bias_norm2s getindex.(bias_vecs, 1) getindex.(
	bias_vecs,
	2,
)]
writedlm("results/tsp_mse_unprojected.txt", data)
