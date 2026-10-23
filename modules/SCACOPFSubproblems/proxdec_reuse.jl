
# Reusable JuMP models for ProxDec.
#
# Included by SCACOPFSubproblems.jl after CoreJuGO
# and the existing auxiliary files.

# ----------------------------------------------------------
# Cached model contexts
# ----------------------------------------------------------

struct ProxDecMasterContext{Q, F, R}
    model::JuMP.Model
    vars::Vector{JuMP.VariableRef}
    p_g::Vector{JuMP.VariableRef}
    base_objective::Q
    production_cost::F
    base_penalty::R
end

struct ProxDecContingencyContext{C, P, R, Q, K}
    model::JuMP.Model
    vars::Vector{JuMP.VariableRef}
    p_g0::Vector{JuMP.VariableRef}
    p_gk::Vector{JuMP.VariableRef}
    aux::Vector{JuMP.VariableRef}
    coupling::C
    penalty::P
    regularizers::R
    base_objective::Q
    kappa_parameter::K
end

# ----------------------------------------------------------
# Contingency data conversion
# ----------------------------------------------------------

# Map original element identifiers from psd.K to model-local
# row indices. Supports multiple outages per contingency.
function proxdec_contingency(psd::SCACOPFdata, k::Int)

    1 <= k <= nrow(psd.K) || throw(BoundsError(psd.K, k))

    types = psd.K[k, :ConType]
    ids   = psd.K[k, :IDout]
    types = types isa AbstractVector ? types : [types]
    ids   = ids isa AbstractVector ? ids : [ids]

    length(types) == length(ids) || error("Inconsistent contingency data for k=$k")

    gen_out = Int[]
    lin_out = Int[]
    trf_out = Int[]

    for (typ, id) in zip(types, ids)
        if typ == :Generator
            row = findfirst(x -> x == id, psd.G[!, :Generator])
            isnothing(row) && error("Generator outage ID $id not found")
            push!(gen_out, row)
        elseif typ == :Line
            row = findfirst(x -> x == id, psd.L[!, :Line])
            isnothing(row) && error("Line outage ID $id not found")
            push!(lin_out, row)
        elseif typ == :Transformer
            row = findfirst(x -> x == id, psd.T[!, :Transformer])
            isnothing(row) && error("Transformer outage ID $id not found")
            push!(trf_out, row)
        else
            error("Unsupported outage type $typ")
        end
    end
    return GenericContingency(gen_out, lin_out, trf_out)
end

# ----------------------------------------------------------
# Solver utilities
# ----------------------------------------------------------

function _proxdec_warm_start!(model::JuMP.Model)
    JuMP.has_values(model) || return nothing

    vars = JuMP.all_variables(model)
    vals = JuMP.value.(vars)

    # Save previous solution as primal starting values.
    for (v, x) in zip(vars, vals)
        if isfinite(x)
            JuMP.set_start_value(v, x)
        end
    end
    return nothing
end

function _proxdec_check_status(model::JuMP.Model)
    ts = JuMP.termination_status(model)
    ps = JuMP.primal_status(model)

    valid_termination = ts in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)
    valid_primal = ps in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
    
    if !(valid_termination && valid_primal)
        error("ProxDec NLP failed: " * "termination=$ts, primal=$ps")
    end
    return nothing
end

# ----------------------------------------------------------
# Master model helpers
# ----------------------------------------------------------

# Positions of generator active-power variables in the
# complete master-state vector.
function proxdec_master_dispatch_indices(ctx::ProxDecMasterContext)
    indices = Dict(v => j for (j, v) in enumerate(ctx.vars))
    return [indices[v] for v in ctx.p_g]
end

# Solve:
#   min f0(y) + g'*(y-xcenter)
#       + alpha*||y-xcenter||^2
#   s.t. base ACOPF constraints
#
# No network constraints or JuMP variables are recreated.
function reoptimize_proxdec_master!(ctx::ProxDecMasterContext,
                                    xcenter::AbstractVector{<:Real},
                                    g::AbstractVector{<:Real},
                                    alpha::Real;
                                    warm_start::Bool=true)
    n = length(ctx.vars)
    length(xcenter) == n || throw(DimensionMismatch("Invalid master center"))
    length(g) == n || throw(DimensionMismatch("Invalid master gradient"))
    isfinite(alpha) && alpha > 0 || error("alpha must be positive and finite")
    all(isfinite, xcenter) || error("Nonfinite master center")
    all(isfinite, g) || error("Nonfinite master gradient")

    m = ctx.model

    # This must happen BEFORE changing the objective.
    warm_start && _proxdec_warm_start!(m)
    obj = copy(ctx.base_objective)
    for j in eachindex(ctx.vars)
        v = ctx.vars[j]
        # alpha * y[j]^2
        JuMP.add_to_expression!(obj, Float64(alpha), v, v)
        # (g[j] - 2*alpha*xcenter[j]) * y[j]
        coeff = g[j] - 2 * alpha * xcenter[j]

        JuMP.add_to_expression!(obj, Float64(coeff), v)
    end

    JuMP.set_objective_function(m, obj)
    JuMP.optimize!(m)

    _proxdec_check_status(m)

    y = JuMP.value.(ctx.vars)

    # The base-objective expression excludes ProxDec terms,
    # and is the authoritative smooth objective.
    f0 = JuMP.value(ctx.base_objective)

    step = y .- xcenter

    master_value = f0 + sum(g[j] * step[j] for j in eachindex(step)) +
                   alpha * sum(abs2, step)

    return (x = y, base_value = f0, master_value = master_value,
            objective = JuMP.objective_value(m), step_inf = maximum(abs.(step)),
            status = JuMP.termination_status(m))
end

# ----------------------------------------------------------
# Contingency proximal oracle
# ----------------------------------------------------------

# Solves
#
#   min_z,y omega*P_k(y) +
#           (1/(2lambda))*||z-pcenter||^2
#
# subject to contingency ACOPF and corrective coupling.
#
# The cached JuMP model stores the unweighted objective:
#
#   P_k(y) + kappa*||z-pcenter||^2,
#
# where kappa = 1/(2lambda*omega).
function reoptimize_proxdec_contingency!(ctx::ProxDecContingencyContext,
                                         pcenter::AbstractVector{<:Real},
                                         lambda::Real,
                                         omega::Real;
                                         warm_start::Bool=true)
    ng = length(ctx.p_g0)

    length(pcenter) == ng || throw(DimensionMismatch("Invalid generator-dispatch center"))
    isfinite(lambda) && lambda > 0 || error("lambda must be positive and finite")
    isfinite(omega) && omega > 0 || error("omega must be positive and finite")

    all(isfinite, pcenter) || error("Nonfinite proximal center")

    kappa = 1.0 / (2.0 * lambda * omega)
    isfinite(kappa) || error("Nonfinite proximal coefficient")

    m = ctx.model

    # Warm start from the previous optimization result.
    warm_start && _proxdec_warm_start!(m)

    # Update p_g0[g] - aux[g] == pcenter[g].
    for j in 1:ng
        JuMP.set_normalized_rhs(ctx.coupling[j], Float64(pcenter[j]))

        # Keep the auxiliary starting point consistent with the updated center.
        zstart = JuMP.start_value(ctx.p_g0[j])

        if !isnothing(zstart) && isfinite(zstart)
            JuMP.set_start_value(ctx.aux[j], zstart - pcenter[j])
        end
    end

    # Update the proximal coefficient.
    if !isnothing(ctx.kappa_parameter)
        # Huber-like nonlinear objective.
        JuMP.set_value(ctx.kappa_parameter, kappa)
    else
        # Entirely quadratic objective.
        obj = copy(ctx.base_objective)

        for a in ctx.aux
            JuMP.add_to_expression!(obj, kappa, a, a)
        end

        JuMP.set_objective_function(m, obj)
    end

    JuMP.optimize!(m)
    _proxdec_check_status(m)

    # Proximal copy of base-case generator dispatch.
    z = JuMP.value.(ctx.p_g0)

    # Corrective contingency generator dispatch.
    pgk = JuMP.value.(ctx.p_gk)

    # Coupling displacement.
    aux = JuMP.value.(ctx.aux)
    displacement = z .- pcenter

    coupling_residual = maximum(abs.(aux .- displacement))

    # Unregularized contingency objective.
    penalty = JuMP.value(ctx.penalty)

    regularization = sum(JuMP.value(expr) for expr in ctx.regularizers)

    recourse_cost = penalty + regularization

    # Weighted Moreau-envelope quantities.
    value = omega * recourse_cost

    envelope = value + sum(abs2, displacement) / (2.0 * lambda)

    grad_p = (pcenter .- z) ./ lambda

    # The JuMP objective is unweighted, so the envelope
    # should equal omega * the solved objective.
    solver_objective = JuMP.objective_value(m)

    consistency_error = abs(envelope - omega * solver_objective)

    return (z = z, p_gk = pgk, displacement = displacement, value = value,
            envelope = envelope, grad_p = grad_p, penalty = penalty,
            regularization = regularization, kappa = kappa,
            solver_objective = solver_objective, coupling_residual = coupling_residual,
            consistency_error = consistency_error,
            status = JuMP.termination_status(m))
end
