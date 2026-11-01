# Shared library setup.  env LIB: target .db
set target_library [list $env(LIB)]
set synthetic_library [list dw_foundation.sldb]
set link_library [concat * $target_library $synthetic_library]

proc zero_wire_load {} {
    set ::auto_wire_load_selection false
    set_load 0 [get_nets -hierarchical *]
}

proc load_netlist {top} {
    read_ddc netlist/mapped.ddc
    current_design $top
    link
    zero_wire_load
}

proc annotate {file inst} {
    reset_switching_activity
    return [catch {read_saif -input $file -instance_name $inst -auto_map_names}]
}
