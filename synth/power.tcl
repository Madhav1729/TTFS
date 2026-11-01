# Hierarchical power per SAIF, zero wire load.  env LIB TOP OUT SAIFS ("tag,file" ...)
source [file join [file dirname [info script]] common.tcl]
set top $env(TOP)
file mkdir $env(OUT)
load_netlist $top

foreach item [split $env(SAIFS)] {
    if {$item eq ""} continue
    lassign [split $item ","] tag file
    if {[annotate $file $top]} { puts "SAIF_FAILED $tag"; continue }
    report_power -hierarchy -nosplit > $env(OUT)/$tag.rpt
    puts "SAIF_DONE $tag"
}
exit
