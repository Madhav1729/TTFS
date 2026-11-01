# Synthesis with clock gating, hierarchy kept.  env LIB TOP PARAMS SRCS PERIOD
source [file join [file dirname [info script]] common.tcl]
set top $env(TOP)
set period $env(PERIOD)
file mkdir reports netlist

analyze -format sverilog [split $env(SRCS)]
elaborate $top -parameters $env(PARAMS)
if {[get_object_name [current_design]] ne $top} { rename_design [current_design] $top }
current_design $top
link

create_clock -name clk -period $period [get_ports clk]
set_clock_uncertainty [expr {0.01 * $period}] [get_clocks clk]
set_input_delay [expr {0.05 * $period}] -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay [expr {0.05 * $period}] -clock clk [all_outputs]
set_load 0.01 [all_outputs]
set_max_area 0
set_clock_gating_style -sequential_cell latch -minimum_bitwidth 3

compile -map_effort medium -gate_clock
change_names -rules verilog -hierarchy
report_area -hierarchy -nosplit > reports/area.rpt
report_clock_gating -nosplit > reports/clock_gating.rpt
report_timing -max_paths 5 -nosplit > reports/timing.rpt
write -format ddc -hierarchy -output netlist/mapped.ddc
exit
