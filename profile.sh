###
### vtune profiling script
###

root=`pwd`
#type="hotspots"
type="advanced-hotspots"
#type="memory-access"
#type="hpc-performance"

/opt/intel/vtune_amplifier/bin64/amplxe-cl \
    -collect $type \
    -knob analyze-openmp=true \
    -knob sampling-interval=10 \
    --resume-after 2 -d 20 \
    -- $root/run.sh $1

