
using Test
using JuMP
using Ipopt

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "modules"))
using SCACOPFSubproblems

const MOI = JuMP.MOI

raw = joinpath(@__DIR__, "examples/500bus/case.raw")
rop = joinpath(@__DIR__, "examples/500bus/case.rop")
con = joinpath(@__DIR__, "examples/500bus/case-3conting.con")

println("Loading SC-ACOPF data ...")

psd = SCACOPFdata(raw_filename=raw, rop_filename=rop, con_filename=con)

@show size(psd.N, 1)
@show size(psd.G, 1)
@show size(psd.K, 1)

@test size(psd.K, 1) == 3

opt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "max_iter" => 1000, "tol" => 1e-7)

# ------------------------------------------------------
# 1. Original base ACOPF
# ------------------------------------------------------

println("\nSolving original base ACOPF ...")

base, base_model = solve_basecase(psd, opt)

@test JuMP.has_values(base_model)
@test isfinite(base.base_cost)

println("Base objective: ", base.base_cost)

# ------------------------------------------------------
# 2. Reusable master
# ------------------------------------------------------

println("\nBuilding ProxDec master ...")

master = solve_basecase(psd, opt; previous_solution=base, build_for_proxdec=true)

master_model = master.model
nvars = JuMP.num_variables(master_model)

xcenter = Float64[isnothing(JuMP.start_value(v)) ? 0.0 : JuMP.start_value(v) for v in master.vars]

grad = zeros(length(xcenter))
alpha = 1.0

println("First master solve ...")

t1 = @elapsed r1 = reoptimize_proxdec_master!(master, xcenter, grad, alpha)

@test isfinite(r1.master_value)
@test length(r1.x) == length(xcenter)

println("Second master solve ...")

# Change the gradient only in generator coordinates.
dispatch_idx = proxdec_master_dispatch_indices(master)

grad2 = copy(grad)
grad2[dispatch_idx] .= 1.0e-3

t2 = @elapsed r2 = reoptimize_proxdec_master!(master, r1.x, grad2, alpha)

@test isfinite(r2.master_value)
@test master.model === master_model
@test JuMP.num_variables(master.model) == nvars

println("Master times: ", t1, " / ", t2)
println("Master objectives: ", r1.master_value, " / ", r2.master_value)

# ------------------------------------------------------
# 3. Reusable contingency
# ------------------------------------------------------

println("\nBuilding ProxDec contingency ...")

lambda = 1.0e-3
omega = (1.0 - psd.delta) / size(psd.K, 1)

ctx = solve_contingency(psd, 1, base, opt;
                        quadratic_relaxation_k= 1.0 / (2.0 * lambda * omega),
                        use_huber_like_penalty=true, build_for_proxdec=true)

cont_model = ctx.model
cont_nvars = JuMP.num_variables(cont_model)

println("First proximal solve ...")

t3 = @elapsed p1 = reoptimize_proxdec_contingency!(ctx, base.p_g, lambda, omega)

@test isfinite(p1.envelope)
@test all(isfinite, p1.grad_p)

@test isapprox(p1.envelope, omega * p1.solver_objective; rtol=1e-6, atol=1e-5)

@test p1.coupling_residual < 1e-5

println("Second proximal solve, changed lambda ...")

t4 = @elapsed p2 = reoptimize_proxdec_contingency!(ctx, base.p_g, 0.8 * lambda, omega)

@test isfinite(p2.envelope)

@test isapprox(p2.envelope, omega * p2.solver_objective; rtol=1e-6, atol=1e-5)

@test ctx.model === cont_model
@test JuMP.num_variables(ctx.model) == cont_nvars

println("Contingency times: ", t3, " / ", t4)

println("Proximal displacement: ", maximum(abs.(p1.displacement)))

println("Envelope: ", p1.envelope)
println("Gradient infinity norm: ", maximum(abs.(p1.grad_p)))

println("\nProxDec reusable-model smoke test passed.")
