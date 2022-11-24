include("operators.jl")
# Example

# 1d derivative
dx = 0.1
x = 0:dx:.5
y = x .^ 2
d = Del((dx,))
d(y)

"""
We use a central difference stencil cut off at both boundaries. To enforce C1 continuity and same output length, the boundary derivative values are repeated from the nearest interior point.
6-element Vector{Float64}:
 0.2
 0.2
 0.4
 0.6
 0.8
 0.8
"""

# 2d gradient
dy = dx = 0.1
a = [x^2 + y^2 for x in 0:dx:0.5, y in 0:dy:0.5]
▽ = Del((dx, dy))
grad_a = ▽(a)

"""
6×6 Matrix{SVector{2, Float64}}:
 [0.0, 0.0]  [0.0, 0.2]  [0.2, 0.4]  [0.2, 0.6]  [0.2, 0.8]  [0.2, 0.8]
 [0.2, 0.0]  [0.2, 0.2]  [0.2, 0.4]  [0.2, 0.6]  [0.2, 0.8]  [0.2, 0.8]
 [0.4, 0.2]  [0.4, 0.2]  [0.4, 0.4]  [0.4, 0.6]  [0.4, 0.8]  [0.4, 0.8]
 [0.6, 0.2]  [0.6, 0.2]  [0.6, 0.4]  [0.6, 0.6]  [0.6, 0.8]  [0.6, 0.8]
 [0.8, 0.2]  [0.8, 0.2]  [0.8, 0.4]  [0.8, 0.6]  [0.8, 0.8]  [0.8, 1.0]
 [0.8, 0.2]  [0.8, 0.2]  [0.8, 0.4]  [0.8, 0.6]  [1.0, 0.8]  [1.0, 1.0]
 """
 
 # 2d divergence
 dy = dx = 0.1
 a = [[2x,3y] for x in 0:dx:0.5, y in 0:dy:0.5]
 ▽ = Del((dx, dy))
▽ ⋅ (a)
#=
6×6 Matrix{Float64}:
 5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0
=#

# 2d Example
dy = dx = 0.1
a = [x^2 + y^2 for x in 0:dx:0.5, y in 0:dy:0.5]
▽2 = Lap((dx, dy))
▽2(a)

#=
6×6 Matrix{Float64}:
 16.0  16.0  16.0  16.0  16.0  16.0
 16.0  16.0  16.0  16.0  16.0  16.0
 16.0  16.0  16.0  16.0  16.0  16.0
 16.0  16.0  16.0  16.0  16.0  16.0
 16.0  16.0  16.0  16.0  16.0  16.0
 16.0  16.0  16.0  16.0  16.0  16.0
=#

dx = dy = 0.1
resolutions = (dx, dy)
rmin = 1e-9
rmax = 0.2
ϕ = Op(r -> 1 / r, rmax, resolutions; rmin)
E = Op(r -> 1 / r^2, rmax, resolutions; rmin, l=1)
g = Grid(resolutions,)
a = zeros(5, 5)
a[3, 3] = 1 / g.dv
ϕ(a)
#=
5×5 Matrix{Float64}:
 0.0   0.0       5.0   0.0      0.0
 0.0   7.07107  10.0   7.07107  0.0
 5.0  10.0       0.0  10.0      5.0
 0.0   7.07107  10.0   7.07107  0.0
 0.0   0.0       5.0   0.0      0.0
=#

E(a)
#=
5×5 Matrix{SVector{2, Float64}}:
 [0.0, 0.0]    [0.0, 0.0]            [-25.0, 0.0]   [0.0, 0.0]           [0.0, 0.0]
 [0.0, 0.0]    [-35.3553, -35.3553]  [-100.0, 0.0]  [-35.3553, 35.3553]  [0.0, 0.0]
 [0.0, -25.0]  [0.0, -100.0]         [0.0, 0.0]     [0.0, 100.0]         [0.0, 25.0]
 [0.0, 0.0]    [35.3553, -35.3553]   [100.0, 0.0]   [35.3553, 35.3553]   [0.0, 0.0]
 [0.0, 0.0]    [0.0, 0.0]            [25.0, 0.0]    [0.0, 0.0]           [0.0, 0.0]
=#

# interpolation


# particle mesh interpolation
dx = dy = dz = 0.1
g = Grid((dx, dy, dz))
a = zeros(5, 5, 5)
v = 1.0
place!(a, g, (0.2, 0.2, 0.2), v)

a[g, 0.2, 0.2, 0.2]
# 1000
v / g.dv
# 1000