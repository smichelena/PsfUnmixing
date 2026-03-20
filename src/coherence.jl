using LinearAlgebra

function mu(grid::AbstractVector, f::Function, g::Function,
	u::Float64, v::Float64, Δ::Float64; supremum_grid_samples=100)::Float64
	μ(u, v, Δ) = abs(dot(f(u, grid), g(v, grid .- Δ)))
	if Δ == 0.0
		return μ(u, v, 0.0) 
	else
		δ = range(Δ, 3Δ, length=supremum_grid_samples)
		return maximum(δ .|> τ -> μ(u,v,τ))
	end
end

function coherence(grid::AbstractVector, f::Function, g::Function,
	u::Float64, v::Float64, Δ::Float64; M=100, supremum_grid_samples=100)::Float64

	μ(u, v, δ) = mu(grid, f, g, u, v, δ, supremum_grid_samples = supremum_grid_samples)

	return sum(μ(u, v, m * Δ) for m ∈ -M:M)
end

function interference(
	M::Int,
	g::Function,
	u::Float64,
	Δ::Float64,
)::Float64
	res = ceil(Int, 100 * (2 / u) + 1)
	δ(τ) = range(τ, τ + 3 * Δ, res)
	sups = [maximum(abs.(g.(u, δ(abs(m) * Δ)))) for m in -M:M]
	return sum(sups)
end
