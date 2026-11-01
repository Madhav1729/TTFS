# PASC-IF stages as PC-cycle multicycle paths (env PC, default 8 at T = 15).
set pc [expr {[info exists env(PC)] && $env(PC) ne "" ? $env(PC) : 8}]
set ends [get_cells -hierarchical * -filter "@is_sequential == true && (@full_name =~ *pasc/r_* || @full_name =~ *pasc/spikes_reg*)"]
puts "APEX multicycle endpoints: [sizeof_collection $ends] (PC = $pc)"
set_multicycle_path $pc -setup -to $ends
set_multicycle_path [expr {$pc - 1}] -hold -to $ends
