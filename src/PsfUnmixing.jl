module PsfUnmixing

include("coherence.jl")

export mu, coherence, interference

include("gauss.jl")

export g_0, g_1, g_2, g_0_n, g_1_n, g_2_n, u_g, s_g

include("lorentz.jl")

export l_0_n, l_1_n, l_2_n, u_l, s_l, l_0, l_1, l_2

include("laplace.jl")

export laplace_0, laplace_1, laplace_2, laplace_3, laplace_0_n, laplace_1_n, laplace_2_n, laplace_3_n, u_laplace, s_laplace

include("model.jl")

export generate_spike_groups, min_separation, single_block, multi_block

include("basins.jl")

export constants_unprojected, constants_projected,
	constants_unprojected_partial, extreme_eigenvalues, lipschitz_manifold_map,
	lipschitz_coherence, lipschitz_total_coherence

include("monte_carlo.jl")

export monte_carlo_unprojected, add_noise_snr, random_perturbation,
	run_solver_unprojected, monte_carlo_projected,
	monte_carlo_RMSE_projected, random_perturbation_projected,
	gradient_descent_projected, loss_projected, block_diag

end # module PsfUnmixing
