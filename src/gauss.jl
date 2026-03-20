# Gauss kernel, free geometry
g_0(u, t) = 1 / (sqrt(2π) * u) * exp.(-(t .^ 2) ./ (2u^2))

g_1(u, t) =
	(1 / (sqrt(2π) * u^2)) * ((t .^ 2) ./ u^2 .- 1) .* exp.(-(t .^ 2) ./ (2u^2))

g_2(u, t) =
	(2u^4 .- 5u^2 * t .^ 2 .+ t .^ 4) ./ (sqrt(2π) * u^7) .*
	exp.(-(t .^ 2) ./ (2u^2))

# Gauss kernel, natural geometry

C = sqrt(3 / (8 * sqrt(π)))

# Arc-length parametrization
s_g(u, umin) = 2C * (1 / sqrt(umin) - 1 / sqrt(u))
u_g(s, umin) = (1 / sqrt(umin) - s / (2C))^(-2)
du_ds(s, umin) = u_g(s, umin)^(3 / 2) / C
du_ds_2(s, umin) = 3u_g(s, umin)^(4 / 2) / (2C^2)

g_0_n(s, t; umin = 1e-4) = g_0(u_g(s, umin), t)

g_1_n(s, t; umin = 1e-4) = du_ds(s, umin) * g_1(u_g(s, umin), t)

g_2_n(s, t; umin = 1e-4) =
	du_ds_2(s, umin) * g_1(u_g(s, umin), t) + (du_ds(s, umin))^2 * g_2(u_g(s, umin), t)
