# Worst paths into the PASC-IF stage registers, to check the multicycle count.
foreach pat {*pasc/r_m1_reg* *pasc/r_s2_reg* *pasc/spikes_reg*} {
    set e [get_cells -hierarchical * -filter "@is_sequential == true && @full_name =~ $pat"]
    if {[sizeof_collection $e] > 0} {
        set d [get_pins -of_objects $e -filter "@direction == in && @name != CK"]
        report_timing -to $d -max_paths 1 -nosplit >> reports/timing_pasc_stages.rpt
    }
}
