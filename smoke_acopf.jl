
using JuMP, Ipopt

push!(LOAD_PATH, joinpath(@__DIR__, "modules"))
using SCACOPFSubproblems

case_dir = joinpath(@__DIR__, "examples", "500bus")

# Load RAW + ROP only, avoiding the contingency path.
println("Loading base-case grid ...")
psd = SCACOPFdata(raw_filename=joinpath(case_dir, "case.raw"), rop_filename=joinpath(case_dir, "case.rop"))

println("Buses: ", size(psd.N, 1))
println("Generators: ", size(psd.G, 1))
println("Lines: ", size(psd.L, 1))
println("Transformers: ", size(psd.T, 1))

opt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 5, "max_iter" => 500, "tol" => 1e-6)

println("Solving base-case ACOPF ...")

base, model = solve_basecase(psd, opt)

println("\n=== ACOPF results ===")
println("Termination: ", termination_status(model))
println("Primal status: ", primal_status(model))
println("Objective: ", base.base_cost)
println("Generator dispatch range (pu): ", extrema(base.p_g))

@assert has_values(model)
@assert isfinite(base.base_cost)
@assert all(isfinite, base.p_g)

println("Base-case ACOPF smoke test passed.")
