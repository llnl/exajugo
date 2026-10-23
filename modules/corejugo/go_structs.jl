# constants

const DELTA = 0.5

# define structure to hold SCACOPF instance data

struct SCACOPFdata
    
    # data frames holding instance information
    MVAbase::Float64
    N::DataFrame
    L::DataFrame
    T::DataFrame
    SSh::DataFrame
    G::DataFrame
    K::DataFrame
    P::DataFrame
    delta::Float64
    
    # relational indexes between power system elements
    L_Nidx::Array{Int,2}
    T_Nidx::Array{Int,2}
    SSh_Nidx::Vector{Int}
    G_Nidx::Vector{Int}
    Lidxn::Vector{Vector{Int}}
    Lin::Vector{Vector{Int}}
    Tidxn::Vector{Vector{Int}}
    Tin::Vector{Vector{Int}}
    SShn::Vector{Vector{Int}}
    Gn::Vector{Vector{Int}}
    K_outidx::Vector{Int}
    RefBus::Int
    Con_RefBus::Vector{Vector{Int}}
    
    # generator cost in epigraph form
    G_epicost_slope::Vector{Vector{Float64}}
    G_epicost_intercept::Vector{Vector{Float64}}
    
    # quadratic approximation coefficients of penalties in P: a*s^2 + b*x
    a::Dict{Symbol, Float64}
    b::Dict{Symbol, Float64}
    
    # tables/data in original format -> needed for solution writing
    generators::DataFrame
    cont_labels::Vector{String}
    
    # constructor from data frames
    function SCACOPFdata(MVAbase::Float64, N::DataFrame, L::DataFrame,
                         T::DataFrame, SSh::DataFrame, G::DataFrame,
                         K::DataFrame, P::DataFrame, generators::DataFrame,
                         contingencies::DataFrame)
        makeup_ramp_rates!(G)
        L_Nidx, T_Nidx, SSh_Nidx, G_Nidx, Lidxn, Lin, Tidxn, Tin, SShn,
            Gn, K_outidx = indexsets(N, L, T, SSh, G, K)
        if nrow(G) > 0
            RefBus = G_Nidx[argmax(G[!,:Pub])]
        else
            RefBus = 1
        end
        Con_RefBus = make_contingency_RefBuses(contingencies, G, N, L, T, G_Nidx)

        if G[1,:CTYP] == 1
            # Not required with polynomial cost function
            G_epicost_slope = Vector{Vector{Float64}}[]
            G_epicost_intercept = Vector{Vector{Float64}}[]
        else
            G_epicost_slope = Vector{Vector{Float64}}(undef, nrow(G))
            G_epicost_intercept = Vector{Vector{Float64}}(undef, nrow(G))
            for i = 1:nrow(G)
                G_epicost_slope[i], G_epicost_intercept[i] = epicost_params(G, i)
            end
        end
        a = Dict{Symbol, Float64}()
        b = Dict{Symbol, Float64}()
        for i = 1:nrow(P)
            a[P[i,:Slack]], b[P[i,:Slack]] = quadcoeffs(P, i)
        end
        gens_identifiers = generators[!,Symbol[:I,:ID]]
        cont_labels = contingencies[!,:LABEL]
        return new(MVAbase, N, L, T, SSh, G, K, P, DELTA,
                   L_Nidx, T_Nidx, SSh_Nidx, G_Nidx, Lidxn, Lin, Tidxn,
                   Tin, SShn, Gn, K_outidx, RefBus, Con_RefBus,
                   G_epicost_slope, G_epicost_intercept, a, b,
                   gens_identifiers, cont_labels)
    end
    
    # constructor from SCACOPF instance folder or RAW file
    function SCACOPFdata(dir::Union{AbstractString, Nothing}=nothing;
                         maxnup::Int=3,
                         raw_filename::Union{AbstractString, Nothing}=nothing,
                         rop_filename::Union{AbstractString, Nothing}=nothing,
                         con_filename::Union{AbstractString, Nothing}=nothing,
                         enforce_bounds_on_x0::Bool=true)
        isnothing(dir) || isnothing(raw_filename) ||
            throw(ArgumentError("either dir or raw_filename [and rop_filename] must be specified"))
        
        if isnothing(raw_filename)
            isnothing(rop_filename) ||
                error("rop_filename may only be specified if raw_filename is specified")
            MVAbase, buses, loads, fixedbusshunts, generators, ntbranches,
                tbranches, switchedshunts, generatordsp, activedsptables,
                costcurves, governorresponse, contingencies =
                readinstance(instancefilenames(dir, maxnup)...)
            N, L, T, SSh, G, K, P =
                GOfmt2params(MVAbase, buses, loads, fixedbusshunts, generators,
                             ntbranches, tbranches, switchedshunts, generatordsp, 
                             activedsptables, costcurves, governorresponse,
                             contingencies, enforce_bounds_on_x0=enforce_bounds_on_x0)
            return SCACOPFdata(MVAbase, N, L, T, SSh, G, K, P,
                               generators, contingencies)
        elseif !isnothing(raw_filename)
            MVAbase, buses, loads, fixedbusshunts, generators, ntbranches,
                tbranches, switchedshunts = readRAW(raw_filename)
            if !isnothing(rop_filename)
                generatordsp, activedsptables, costcurves = readROP(rop_filename)
            else
                generatordsp = nothing
                activedsptables = nothing
                costcurves = nothing
            end
            if !isnothing(con_filename)
                contingencies = readCON(con_filename)
                N, L, T, SSh, G, K, P =
                    GOfmt2params(MVAbase, buses, loads, fixedbusshunts, generators,
                                ntbranches, tbranches, switchedshunts, generatordsp, 
                                activedsptables, costcurves, 
                                DataFrame([Int64[], String[], Float64[], Float64[], 
                                            Float64[], Float64[], Float64[]],
                                [:I,:ID,:H,:PMAX,:PMIN,:R,:D]), 
                                contingencies,
                                enforce_bounds_on_x0=enforce_bounds_on_x0)
                return SCACOPFdata(MVAbase, N, L, T, SSh, G, K, P,
                                generators, contingencies)
            else
                N, L, T, SSh, G, K, P =
                    GOfmt2params(MVAbase, buses, loads, fixedbusshunts, generators,
                                ntbranches, tbranches, switchedshunts,
                                generatordsp, activedsptables, costcurves,
                                enforce_bounds_on_x0=enforce_bounds_on_x0)
                return SCACOPFdata(MVAbase, N, L, T, SSh, G, K, P,
                                generators, DataFrame([String[], Symbol[], Contingency[]],
                                                        [:LABEL, :CTYPE, :CON]))
            end
        end
    end
    
end

# function to create the set of reference buses for each contingency

function make_contingency_RefBuses(contingencies::DataFrame, G::DataFrame, N::DataFrame,
                                    L::DataFrame, T::DataFrame, 
                                    G_Nidx::Vector{Int})::Vector{Vector{Int}}
    # Initialize a vector of integer vectors to store the reference bus for each contingency
    Refbuses = Vector{Vector{Int}}(undef, size(contingencies)[1])

    # Get the list of unique generator bus IDs from the power system data
    gen_bus_id = unique(G.Bus)

    # Loop through each contingency case
    for k = 1:length(contingencies.CON)
        confb = Int[]       # From buses
        contb = Int[]       # To buses
        conskt = Vector{Vector{Int64}}()  # Each entry stores a vector of circuit IDs

        for i in 1:length(contingencies.CON[k])
            con = contingencies.CON[k][i]

            if hasproperty(con, :FromBus)
                # Find existing index where (FromBus, ToBus) matches
                idx = findfirst(j -> confb[j] == con.FromBus && contb[j] == con.ToBus, eachindex(confb))

                if isnothing(idx)
                    # First time we see this (FromBus, ToBus) pair → create new entry
                    push!(confb, con.FromBus)
                    push!(contb, con.ToBus)
                    push!(conskt, [parse(Int, con.Ckt)])   # Start new vector
                else
                    # Pair already exists → append to that inner vector
                    push!(conskt[idx], parse(Int, con.Ckt))
                end
            end
        end 

        # Create a new graph representing the network topology.
        # Number of vertices = total number of lines + total number of transformers.
        g = Graph(length(N.Bus))

        # ---------------------------
        # Add transmission line edges
        # ---------------------------
        for i = 1:length(L.From)
            # Find all indices where both FromBus and ToBus match
            idxfb = findall(x -> x == L.From[i], confb)
            idxtb = findall(x -> x == L.To[i], contb)
            idx = intersect(idxfb, idxtb)
            if isempty(idx)
                # No existing edge — add it directly
                add_edge!(g, L.From[i], L.To[i])
            else
                if !(parse(Int, L.CktID[i]) in conskt[idx[:]][1] )
                    add_edge!(g, L.From[i], L.To[i])
                end
            end
        end

        # ---------------------------
        # Add transformer connections
        # ---------------------------
        for i = 1:length(T.From)
            # Find all indices where both FromBus and ToBus match            
            idxfb = findall(x -> x == T.From[i], confb)
            idxtb = findall(x -> x == T.To[i], contb)
            idx = intersect(idxfb, idxtb)
            if isempty(idx)
                # No existing edge — add it directly
                add_edge!(g, T.From[i], T.To[i])
            else
                if !(parse(Int, T.CktID[i]) in conskt[idx[:]][1] )
                    add_edge!(g, T.From[i], T.To[i])
                end            
            end

        end

        # ---------------------------
        # Identify islands (connected components)
        # ---------------------------
        # Each connected component corresponds to an electrical island after the contingency
        islands = connected_components(g)
        islands = filter(x -> length(x) > 1, islands)

        # Initialize the reference bus list for this contingency
        Refbuses[k] = Int[]

        # Loop through each island to determine a reference bus
        for is = 1:length(islands)

            # Find which generator buses belong to this island
            island_gens = intersect(islands[is], gen_bus_id)

            # If no generators exist in this island, choose the first bus as reference
            if length(island_gens) == 0
                push!(Refbuses[k], islands[is][1])
            else
                # Otherwise, find the generator in this island with the highest 'Pub' (likely active power)
                # 1. Find indices in G.Bus corresponding to these generator buses
                all_indices = vcat([findall(x -> x == val, G.Bus) for val in island_gens]...)

                # 2. Push the bus index (mapped through G_Nidx) with the maximum Pub value
                push!(Refbuses[k], G_Nidx[all_indices[argmax(G[all_indices, :Pub])]])
            end
        end
    end
    return Refbuses
end


# function to check solution dimensions

function check_solution_dimensions(psd::SCACOPFdata, 
                                   v_n::Vector{Float64}, theta_n::Vector{Float64},
                                   b_s::Vector{Float64},
                                   p_g::Vector{Float64}, q_g::Vector{Float64})
    if nrow(psd.N) != length(v_n) || length(v_n) != length(theta_n)
        DimensionMismatch()
    end
    if nrow(psd.SSh) != length(b_s)
        DimensionMismatch()
    end
    if nrow(psd.G) != length(p_g) || length(p_g) != length(q_g)
        DimensionMismatch()
    end
    return nothing
end

# define structure to hold base case solution data

abstract type SubproblemSolution end

mutable struct BasecaseSolution <: SubproblemSolution
    
    # hash of SCACOPF instance data (useful for checks)
    psd_hash::UInt64        # psd: power system data
    
    # vectors containing the solution (state variables only)
    v_n::Vector{Float64}
    theta_n::Vector{Float64}
    b_s::Vector{Float64}
    p_g::Vector{Float64}
    q_g::Vector{Float64}
    
    # objective information
    base_cost::Union{Float64, Nothing}
    recourse_cost::Union{Float64, Nothing}
    total_objective::Union{Float64, Nothing}

    # constructor
    function BasecaseSolution(psd::SCACOPFdata, 
                              v_n::Vector{Float64}, theta_n::Vector{Float64},
                              b_s::Vector{Float64},
                              p_g::Vector{Float64}, q_g::Vector{Float64},
                              base_cost::Union{Float64, Nothing},
                              recourse_cost::Union{Float64, Nothing},
                              total_objective::Union{Float64, Nothing}) #added total obj
        check_solution_dimensions(psd, v_n, theta_n, b_s, p_g, q_g)
        return new(hash(psd), v_n, theta_n, b_s, p_g, q_g, base_cost,
                   recourse_cost, total_objective)
    end
    
    # basic constructor (allocate to then fill)
    function BasecaseSolution(psd::SCACOPFdata)
        vec(field::Symbol) = Vector{Float64}(undef, nrow(getfield(psd, field)))
        return new(hash(psd), vec(:N), vec(:N), vec(:SSh), vec(:G), vec(:G),
                   nothing, nothing, nothing)
    end

end

# define structure to specify a general continceny

struct GenericContingency
    
    # members: indices only
    generators_out::Vector{Int}
    lines_out::Vector{Int}
    transformers_out::Vector{Int}
    
    # empty constructor
    function GenericContingency()
        return new(Int[], Int[], Int[])
    end
   
    # constructor from indices
    function GenericContingency(gen_out::Vector{<:Integer},
                                lin_out::Vector{<:Integer},
                                trf_out::Vector{<:Integer})
        return new(gen_out, lin_out, trf_out)
    end
    
    # constructor from system
    function GenericContingency(psd::SCACOPFdata, k::Int)
        k >= 0 && k <= nrow(psd.K) || error("contingency index out of bounds")
        if psd.K[k, :ConType] == :Generator
            return new([psd.K_outidx[k]], Int[], Int[])
        elseif psd.K[k, :ConType] == :Line
            return new(Int[], [psd.K_outidx[k]], Int[])
        else
            @assert psd.K[k, :ConType] == :Transformer "unrecognized contingency type"
            return new(Int[], Int[], [psd.K_outidx[k]])
        end
    end
    
end

function isequal_struct(x::T, y::T)::Bool where {T}
    isstructtype(T) || error("arguments should be structs")
    for f in fieldnames(T)
        getfield(x, f) == getfield(y, f) || return false
    end
    return true
end

# define structure to hold contingency solution data

mutable struct ContingencySolution <: SubproblemSolution
    
    # hash of SCACOPF instance data (useful for checks)
    psd_hash::UInt64
    
    # information to identify contingency
    cont_Kidx::Union{Int, Vector{Int}}               # index in K
    cont_id::Union{Int, Vector{Int}}            # index in contingencies
    cont_idout::Union{Int, Vector{Int}}              # id of the element out
    cont_type::Union{Symbol, Vector{Symbol}}          # contingency type: :Line, :Transformer, :Generator
    
    # alternative generic contingency (for running contingencies not in K)
    cont_alt::Union{GenericContingency, Nothing}
    
    # vectors containing the solution (state variables only)
    v_n::Vector{Float64}
    theta_n::Vector{Float64}
    b_s::Vector{Float64}
    p_g::Vector{Float64}
    q_g::Vector{Float64}
    delta::Float64
    
    # objective information: value and gradient/subgradient
    cont_cost::Float64
    cont_grad::Union{Nothing, Vector{Float64}}
    
    # constructor for regular contingencies
    function ContingencySolution(psd::SCACOPFdata, k::Int,
                                 v_n::Vector{Float64}, theta_n::Vector{Float64},
                                 b_s::Vector{Float64},
                                 p_g::Vector{Float64}, q_g::Vector{Float64},
                                 delta::Float64, cont_cost::Float64,
                                 cont_grad::Union{Nothing, Vector{Float64}}=nothing)
        check_solution_dimensions(psd, v_n, theta_n, b_s, p_g, q_g)
        return new(hash(psd),
                   k, psd.K[k,:Contingency], psd.K[k,:IDout], psd.K[k,:ConType], nothing,
                   v_n, theta_n, b_s, p_g, q_g, delta,
                   cont_cost, cont_grad)
    end
    
    # constructor for generic contingencies
    function ContingencySolution(psd::SCACOPFdata, con::GenericContingency,
                                 v_n::Vector{Float64}, theta_n::Vector{Float64},
                                 b_s::Vector{Float64},
                                 p_g::Vector{Float64}, q_g::Vector{Float64},
                                 delta::Float64, cont_cost::Float64,
                                 cont_grad::Union{Nothing, Vector{Float64}}=nothing)
        check_solution_dimensions(psd, v_n, theta_n, b_s, p_g, q_g)
        return new(hash(psd),
                   0, 0, 0, :None, con,
                   v_n, theta_n, b_s, p_g, q_g, delta,
                   cont_cost, cont_grad)
    end
    
end

# define structure to hold SCACOPF solution data

mutable struct SCACOPFsolution
    
    # hash of SCACOPF instance data (useful for checks)
    psd_hash::UInt64
    
    # solutions to base case and contingencies (maybe not all)
    basecase::BasecaseSolution
    contingency::Vector{ContingencySolution}
    
    # constructor
    function SCACOPFsolution(psd::SCACOPFdata, basecase::BasecaseSolution)
        if hash(psd) != basecase.psd_hash
            error("base case solution does not correspond to power system data.")
        end
        this = new()
        this.psd_hash = basecase.psd_hash
        this.basecase = basecase
        this.contingency = ContingencySolution[]
        return this
    end
    
end

function add_contingency_solution!(scacopfsol::SCACOPFsolution,
                                   consol::ContingencySolution)
    if scacopfsol.psd_hash != consol.psd_hash
            error("contingency solution does not correspond to SCACOPF solution being pushed into.")
    end
    push!(scacopfsol.contingency, consol)
    return nothing
end


## auxiliary (internal) functions

# function for computing indices for efficient formulation

function indexsets(N::DataFrame, L::DataFrame, T::DataFrame, SSh::DataFrame,
                   G::DataFrame, K::Union{DataFrame, Tuple{Symbol, Int}})
    
    L_Nidx = hcat(convert(Vector{Int}, indexin(L[!,:From], N[!,:Bus])),
        convert(Vector{Int}, indexin(L[!,:To], N[!,:Bus])))
    T_Nidx = hcat(convert(Vector{Int}, indexin(T[!,:From], N[!,:Bus])),
        convert(Vector{Int}, indexin(T[!,:To], N[!,:Bus])))
    SSh_Nidx = convert(Vector{Int}, indexin(SSh[!,:Bus], N[!,:Bus]))
    G_Nidx = convert(Vector{Int}, indexin(G[!,:Bus], N[!,:Bus]))
    Lidxn = Vector{Vector{Int}}(undef, size(N, 1))
    Lin = Vector{Vector{Int}}(undef, size(N, 1))
    Tidxn = Vector{Vector{Int}}(undef, size(N, 1))
    Tin = Vector{Vector{Int}}(undef, size(N, 1))
    SShn = Vector{Vector{Int}}(undef, size(N, 1))
    Gn = Vector{Vector{Int}}(undef, size(N, 1))
    for n = 1:size(N, 1)
        Lidxn[n] = Int[]
        Lin[n] = Int[]
        Tidxn[n] = Int[]
        Tin[n] = Int[]
        SShn[n] = Int[]
        Gn[n] = Int[]
    end
    for l = 1:size(L, 1), i=1:2
        push!(Lidxn[L_Nidx[l,i]], l)
        push!(Lin[L_Nidx[l,i]], i)
    end
    for t = 1:size(T, 1), i=1:2
        push!(Tidxn[T_Nidx[t,i]], t)
        push!(Tin[T_Nidx[t,i]], i)
    end
    for s = 1:size(SSh, 1)
        push!(SShn[SSh_Nidx[s]], s)
    end
    for g = 1:size(G, 1)
        push!(Gn[G_Nidx[g]], g)
    end
    if typeof(K) == DataFrame
        K_outidx = Vector{Vector{Int}}(undef, size(K, 1))
        Kgen = findall(K[!,:ConType] .== :Generator)
        Klin = findall(K[!,:ConType] .== :Line)
        Ktra = findall(K[!,:ConType] .== :Transformer)
        K_outidx[Kgen] .= convert(Vector{Int},
                                  indexin(K[Kgen,:IDout], G[!,:Generator]))
        K_outidx[Klin] .= convert(Vector{Int},
                                  indexin(K[Klin,:IDout], L[!,:Line]))
        K_outidx[Ktra] .= convert(Vector{Int},
                                  indexin(K[Ktra,:IDout], T[!,:Transformer]))
    else
        if K[1] == :Generator
            K_outidx = Int(findfirst(G[!,:Generator] .== K[2]))
        elseif K[1] == :Line
            K_outidx = Int(findfirst(L[!,:Line] .== K[2]))
        elseif K[1] == :Transformer
            K_outidx = Int(findfirst(T[!,:Transformer] .== K[2]))
        else
            error("unrecognized contingency type ", K[1])
        end
    end
    
    return L_Nidx, T_Nidx, SSh_Nidx, G_Nidx, Lidxn, Lin, Tidxn, Tin, SShn,
           Gn, K_outidx
    
end

# function to compute quadratic coefficients for peanlty terms

function quadcoeffs(P::DataFrame, idx::Int)
    perm = sortperm(P[idx,:Penalties])
    quantities = view(P[idx,:Quantities], perm)
    penalties = view(P[idx,:Penalties], perm)
    b = penalties[1]
    a = (penalties[2] - b)/(2*quantities[1])
    b >= 0 || error("penalty function decreasing at zero for "*
                    string(P[idx,:Slack]))
    a >= 0 || error("non-convex penalty function for "*string(P[idx,:Slack]))
    return a, b
end

# function to compute epigraph formulation coefficients for generator cost

function epicost_params(G::DataFrame, g::Int)
    
    # get (production, cost) pairs
    p_i = G[g,:CostPi]
    c_i = G[g,:CostCi]
    if !(typeof(p_i) <: Vector{<:Real} && typeof(c_i) <: Vector{<:Real})
        error("incorrect types of cost data for generator I=" * 
              string(G[g,:Bus]) * ", ID=" * string(G[g,:BusUnitNum]) * ".")
    end
    if length(p_i) != length(c_i)
        DimensionMismatch()
    end
    
    # return for trivial cases
    if length(p_i) == 0
        @warn "no cost information for generator I=" * 
              string(G[g,:Bus]) * ", ID=" * string(G[g,:BusUnitNum]) *
              ". Will assume zero."
        return [0.0], [0.0]
    end
    if length(p_i) == 1
        @warn "only one (production, cost) point provided for generator I=" * 
              string(G[g,:Bus]) * ", ID=" * string(G[g,:BusUnitNum]) *
              ". Will assume constant cost."
        return [0.0], c_i
    end
    
    # compute slopes and intercepts, and return
    n_points = length(p_i)
    slope = Vector{Float64}(undef, n_points-1)
    intercept = Vector{Float64}(undef, n_points-1)
    for i = 1:(n_points-1)
        slope[i] = (c_i[i+1] - c_i[i])/(p_i[i+1] - p_i[i])
        intercept[i] = c_i[i] - slope[i]*p_i[i]
    end
    
    # check convexity
    if any(diff(slope) .< 0)
        @warn "nonconvex cost curve provided for generator I=" * 
              string(G[g,:Bus]) * ", ID=" * string(G[g,:BusUnitNum]) * "."
    end
    
    # return slope and intercept
    return slope, intercept
    
end

# function to make up maximum ramp up/down rates from generation data

function makeup_ramp_rates!(G::DataFrame,
                            min_ramp_rate::Float64=0.02,
                            max_ramp_rate::Float64=0.06)::Nothing
    max_ramp_rate > min_ramp_rate && min_ramp_rate >= 0.0 ||
        error("invalid minimum and maximum ramp-rate parameters")
    rng = Random.MersenneTwister(hash(G))   # create random number generator with a fixed seed for given G
    G[:, :RampRate] = rand(rng, nrow(G)) * (max_ramp_rate - min_ramp_rate) .+ min_ramp_rate
    return nothing
end
