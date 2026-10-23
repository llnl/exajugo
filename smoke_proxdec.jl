using Test
using JuMP
using Ipopt
using Printf

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

lambda = 1.0e-3
omega = (1.0 - psd.delta) / size(psd.K, 1)

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

# ----------------------------------------------------------
# 4. Test all contingencies
# ----------------------------------------------------------

@testset "All ProxDec contingencies" begin
    for k in 1:size(psd.K, 1)
        println("Testing contingency $k")
        cctx = solve_contingency(psd, k, base, opt;
                                 quadratic_relaxation_k = 1.0 / (2.0 * lambda * omega),
                                 use_huber_like_penalty = true, build_for_proxdec = true)

        r = reoptimize_proxdec_contingency!(cctx, base.p_g, lambda, omega)

        @test isfinite(r.envelope)
        @test all(isfinite, r.grad_p)
        @test r.coupling_residual < 1e-5
        @test isapprox(r.envelope, omega * r.solver_objective; atol=1e-5, rtol=1e-6)

        println("  Envelope: ", r.envelope)
        println("  Displacement infinity norm: ", maximum(abs.(r.displacement)))
    end
end


# ----------------------------------------------------------
# 5. Test all contingencies and screen critical scenarios
# ----------------------------------------------------------
screening = NamedTuple[]

@testset "All ProxDec contingencies screening" begin
    for k in 1:size(psd.K, 1)
        label = string(psd.K[k, :Label])

        cctx = solve_contingency(psd, k, base, opt;
                                 quadratic_relaxation_k = 1.0 / (2.0 * lambda * omega),
                                 use_huber_like_penalty = true, build_for_proxdec = true)

        elapsed = @elapsed r = reoptimize_proxdec_contingency!(cctx, base.p_g, lambda, omega)

        # Unweighted objective breakdown:
        # P_k + R_k + kappa * ||z - p||^2
        penalty = r.penalty
        reg = r.regularization
        prox_quad = r.kappa * sum(abs2, r.displacement)
        total = penalty + reg + prox_quad

        @test isfinite(penalty)
        @test isfinite(reg)
        @test isfinite(prox_quad)
        @test isfinite(r.envelope)
        @test all(isfinite, r.grad_p)
        @test r.coupling_residual < 1e-5

        @test isapprox(total, r.solver_objective; atol=1e-5, rtol=1e-6)

        @test isapprox(r.envelope, omega * total; atol=1e-5, rtol=1e-6)

        row = (k = k, label = label, penalty = penalty, regularization = reg,
               proximal = prox_quad, total = total, weighted_penalty = omega * penalty,
               envelope = r.envelope, step_inf = maximum(abs.(r.displacement)), time = elapsed)

        push!(screening, row)

        @printf("\nContingency %d: %s\n", k, label)
        @printf("  Technical penalty  P_k       = %14.6e\n", penalty)
        @printf("  Other regularization R_k      = %14.6e\n", reg)
        @printf("  Proximal quadratic            = %14.6e\n", prox_quad)
        @printf("  Total NLP objective           = %14.6e\n", total)
        @printf("  Weighted penalty omega*P_k    = %14.6e\n", omega * penalty)
        @printf("  Weighted envelope             = %14.6e\n", r.envelope)
        @printf("  Proximal displacement inf     = %14.6e\n", row.step_inf)
        @printf("  NLP solve time                = %10.3f s\n", elapsed)
    end
end

# Rank scenarios by technical-violation penalty.
sort!(screening; by = r -> r.penalty, rev = true)

println("\nContingency screening (sorted by penalty)")
@printf("%4s %-45s %12s %12s %12s %12s\n",
        "k", "Label", "Penalty", "Other reg.",
        "Prox quad.", "Envelope")

for r in screening
    @printf("%4d %-45s %12.4e %12.4e %12.4e %12.4e\n",
            r.k, first(r.label, min(length(r.label), 45)),
            r.penalty, r.regularization,
            r.proximal, r.envelope)
end

# ----------------------------------------------------------
# 6. Test standalone contingencies 
# ----------------------------------------------------------

@testset "Standalone fixed-dispatch contingencies" begin
    for k in 1:size(psd.K, 1)
        println("\nSolving fixed-dispatch contingency $k: ", psd.K[k, :Label])

        # Expected model-local outage indices.
        con = proxdec_contingency(psd, k)

        # Original, non-ProxDec solve.
        sol = solve_contingency(psd, k, base, opt; quadratic_relaxation_k = Inf, use_huber_like_penalty = true)

        @test isfinite(sol.cont_cost)
        @test all(isfinite, sol.p_g)
        @test sol.cont_Kidx == k
        @test sol.cont_idout == psd.K[k, :IDout]

        # Outaged generator dispatch must be zero.
        for g in con.generators_out
            @test abs(sol.p_g[g]) <= 1e-6
            @test abs(sol.q_g[g]) <= 1e-6
        end

        println("  Contingency objective: ", sol.cont_cost)
        println("  Outaged generator rows: ", con.generators_out)
        println("  Outaged line rows: ", con.lines_out)
        println("  Outaged transformer rows: ", con.transformers_out)
    end
end

println("\nProxDec reusable-model smoke test passed.")
