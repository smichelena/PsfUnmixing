
function lipschitz_manifold_map(k_1, M, grid, umin, umax)
	u_range = logrange(umin, umax, 10)
	μ(u) = mu(grid, k_1, k_1, u, u, 0.0)
	return sqrt(M * maximum(μ.(u_range)))
end

function finite_diff(f, x::AbstractVector)
	fx = f.(x)
	dx = diff(x)
	df = similar(x)

	# Central differences for interior points
	df[2:end-1] .= (fx[3:end] .- fx[1:end-2]) ./ (x[3:end] .- x[1:end-2])

	# Forward/backward for boundaries
	df[1] = (fx[2] - fx[1]) / dx[1]
	df[end] = (fx[end] - fx[end-1]) / dx[end-1]

	return df
end

# for a single variable simplified case and natural geometry
function lipschitz_coherence(k, Δ, grid, umin, umax)
	u_range = logrange(umin, umax, 10)
	μ(u) = mu(grid, k, k, u, u, Δ)
	∂μ = abs.(finite_diff(μ, u_range))
	return ∂μ[end]
end

# for a single variable simplified case and natural geometry
function lipschitz_total_coherence(k, Δ, grid, umin, umax)
	u_range = logrange(umin, umax, 10)
	C(u) = coherence(10, grid, k, k, u, u, Δ)
	∂μ = abs.(finite_diff(C, u_range))
	return ∂μ[end]
end


# this assumes the natural geometry of a manifold is being using, alongside a relative parametrization
function constants_unprojected(k_0, k_1, k_2, grid, N, L, u_, η_, Δ_, umin, umax)
    M = 10

	C_g_2 = lipschitz_manifold_map(k_1, L, grid, umin, umax)

	C_μ_00 = lipschitz_coherence(k_0, Δ_, grid, umin, umax)
	C_μ_11 = lipschitz_coherence(k_1, Δ_, grid, umin, umax)
	C_μ_22 = lipschitz_coherence(k_2, Δ_, grid, umin, umax)

	C_Δ_00 = lipschitz_total_coherence(k_0, Δ_, grid, umin, umax)
	C_Δ_11 = lipschitz_total_coherence(k_1, Δ_, grid, umin, umax)
	C_Δ_22 = lipschitz_total_coherence(k_2, Δ_, grid, umin, umax)

	C_g_infty = N * C_g_2
	C_μ = maximum([C_μ_00, C_μ_11])
	C_Δ = maximum([C_Δ_00, C_Δ_11, C_Δ_22])

	I_0 = interference(M, k_0, u_, Δ_)
	I_1 = interference(M, k_1, u_, Δ_)
	I_2 = interference(M, k_2, u_, Δ_)

	S_10 = 2 * coherence(M, grid, k_1, k_0, u_, u_, Δ_) + mu(grid, k_1, k_0, u_, u_, 0.0)

	Λ_min_0 = 0.5 * mu(grid, k_0, k_0, u_, u_, 0.0) - 2 * coherence(M, grid, k_0, k_0, u_, u_, Δ_)

	c_2_0 = N^(-1) * (Λ_min_0 - S_10)
	c_2_1 = N^(-1) * (C_μ + 4 * 2 * C_Δ)

	r_0 = L * (η_ * I_2 + I_1)
	r_1 =
		L * (
			η_ * (I_2 * I_0 + C_g_infty * I_1) + C_g_infty * η_^2 * I_2 +
			I_2 * I_0
		)
	r_1_xi = L * (η_ * C_Δ + I_2 + C_Δ)


    return c_2_0, c_2_1, r_0, r_1, r_1_xi
end

function constants_unprojected_partial(k_0, k_1, k_2, grid, N, L, u_, η_min, η_max, Δ_, umin, umax)
    M = 10

	C_g_2 = lipschitz_manifold_map(k_1, L, grid, umin, umax)
	C_g_infty = N * C_g_2

	C_μ = lipschitz_coherence(k_1, Δ_, grid, umin, umax)

	C_Δ_11 = lipschitz_total_coherence(k_1, Δ_, grid, umin, umax)
	C_Δ_22 = lipschitz_total_coherence(k_2, Δ_, grid, umin, umax)

	C_Δ = maximum([C_Δ_11, C_Δ_22])
	
	I_0 = interference(M, k_0, u_, Δ_)
	I_2 = interference(M, k_2, u_, Δ_)

	Λ_min_1 = 0.5 * mu(grid, k_1, k_1, u_, u_, 0.0) - 2 * coherence(M, grid, k_1, k_1, u_, u_, Δ_)

	c_1_0 = N^(-1) * η_min^2 * Λ_min_1
	c_1_1 = N^(-1) * (η_min^2 *(C_μ + 4 * 2 * C_Δ) + η_min*Λ_min_1)

	r_0 = L * η_max * I_2
	r_1 = L * (η_max * I_2 * I_0 + C_g_infty * η_max^2 * I_2 + I_2 * I_0)
	r_1_xi = L * (η_max * C_Δ + I_2 + C_Δ)


    return c_1_0, c_1_1, r_0, r_1, r_1_xi
end

function extreme_eigenvalues(
	k_0, umin, umax, T, grid;
	resolution = 50,  # resolution per axis
)
	us = range(umin, umax, length = resolution)
	λ_min = Inf
	λ_max = -Inf

	for u₁ in us, u₂ in us
		G = multi_block(k_0, [u₁, u₂], T, grid)
		λs = eigvals(Hermitian(transpose(G) * G))
		λ_min = min(λ_min, minimum(λs))
		λ_max = max(λ_max, maximum(λs))
	end

	return λ_min, λ_max
end


function constants_projected(k_0, k_1, k_2, grid, T, u_, Δ_, L, umin, umax)
	M = 10

	N = length(grid)

	C_g_2 = lipschitz_manifold_map(k_1, L, grid, umin, umax)

	C_μ_00 = lipschitz_coherence(k_0, Δ_, grid, umin, umax)
	C_μ_11 = lipschitz_coherence(k_1, Δ_, grid, umin, umax)

	C_Δ_00 = lipschitz_total_coherence(k_0, Δ_, grid, umin, umax)
	C_Δ_11 = lipschitz_total_coherence(k_1, Δ_, grid, umin, umax)
	C_Δ_22 = lipschitz_total_coherence(k_2, Δ_, grid, umin, umax)

	C_μ = maximum([C_μ_00, C_μ_11])
	C_Δ = maximum([C_Δ_00, C_Δ_11, C_Δ_22])

	λ_min_0 = 0.5 * mu(grid, k_0, k_0, u_, u_, 0.0) - coherence(M, grid, k_0, k_0, u_, u_, Δ_)
	λ_min_2 = 0.5 * mu(grid, k_2, k_2, u_, u_, 0.0) - coherence(M, grid, k_2, k_2, u_, u_, Δ_)

	λ_max_0 = mu(grid, k_0, k_0, u_, u_, 0.0) + coherence(M, grid, k_0, k_0, u_, u_, Δ_)
	λ_max_2 = mu(grid, k_2, k_2, u_, u_, 0.0) + coherence(M, grid, k_2, k_2, u_, u_, Δ_)

	Λ_min_0 = 0.5 * mu(grid, k_0, k_0, u_, u_, 0.0) - 2 * coherence(M, grid, k_0, k_0, u_, u_, Δ_)
	Λ_min_1 = 0.5 * mu(grid, k_1, k_1, u_, u_, 0.0) - 2 * coherence(M, grid, k_1, k_1, u_, u_, Δ_)

    λ_m, λ_M = extreme_eigenvalues(k_0,  1e-2, 0.2, T, grid, resolution = 100) # effy stuff with the boundaries

    κ_A = λ_M / λ_m

	α =
		(λ_max_0 * (C_μ + 2 * 2 * C_Δ) + 2 * Λ_min_1 * (C_μ + C_Δ)) / λ_max_0^2

	β =
		(λ_min_0 * (C_μ + C_Δ) + 2 * λ_max_2 * (C_μ + 2 * C_Δ)) /
		(2 * sqrt(λ_min_0^3 * λ_min_2))

	γ =
		(C_g_2 * (1 + 2 * κ_A) * (1 + Λ_min_0) * sqrt(λ_max_2)) /
		(λ_m * sqrt(N * λ_min_0 * Λ_min_0))

    return Λ_min_1 / λ_max_0, sqrt(λ_max_2 / λ_min_0), α, β, γ

end