cd ./flow_analysis/
julia main_flow.jl
cd ../MadrasSokal/
julia "scripts/table.jl"
julia "scripts/plaquette.jl"
julia "scripts/topological_charge.jl"
cd ..

