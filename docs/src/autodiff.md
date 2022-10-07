# Automatic differentiation

Most of the package is compatible with autodiff in `Flux` and `Zygote`. Exceptions:
- may or may not work with FFT convolutions. set `alg = :direct` if encountering autodiff error
- some special `border` options 

An alternative autodiff package is `Enzyme.jl` which works at the LLVM level and can handle limitations in `Zygote` eg array mutation. It's probably the future of autodiff in Julia/C++ but isn't yet mature.