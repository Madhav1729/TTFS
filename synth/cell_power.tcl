# Summed power of the cells matching CELLS (e.g. *collector/*table_*) per SAIF.  env LIB TOP OUT SAIFS CELLS
source [file join [file dirname [info script]] common.tcl]
set top $env(TOP)
file mkdir $env(OUT)
load_netlist $top
set cells [get_cells -hier -filter "full_name=~$env(CELLS)"]

set area 0.0
foreach_in_collection c $cells { set area [expr {$area + [get_attribute $c area]}] }
set fh [open $env(OUT)/area.txt w]; puts $fh "$area [sizeof_collection $cells]"; close $fh

foreach item [split $env(SAIFS)] {
    if {$item eq ""} continue
    lassign [split $item ","] tag file
    if {[annotate $file $top]} { puts "SAIF_FAILED $tag"; continue }
    report_power -cell -nosplit -only $cells > $env(OUT)/$tag.full
    exec grep "^Totals" $env(OUT)/$tag.full > $env(OUT)/$tag.rpt
    file delete $env(OUT)/$tag.full
    puts "SAIF_DONE $tag"
}
exit
