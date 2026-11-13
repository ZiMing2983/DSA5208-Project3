# Toolchain for Atlas jobs: GCC 13.2 + Open MPI 4.1.6. Source it from a job script.
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
[ -f /app1/ebenv ] && source /app1/ebenv >/dev/null 2>&1
module load gompi/2023b
export CG_MACHINE=atlas
