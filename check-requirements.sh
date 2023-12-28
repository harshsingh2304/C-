#!/bin/bash
set -euo pipefail
#
# check-requirements.sh checks all requirements files for each top-level
# convert*.py script.
#
# WARNING: This is quite IO intensive, because a fresh venv is set up for every
# python script. As of 2023-12-22, this writes ~2.7GB of data. An adequately
# sized tmpfs /tmp or ramdisk is recommended if running this frequently.
#
# usage:    ./check-requirements.sh [<working_dir>]
#           ./check-requirements.sh nocleanup [<working_dir>]
#
# where:
#           - <working_dir> is a directory that can be used as the base for
#               setting up the venvs. Defaults to `/tmp`.
#           - 'nocleanup' as the first argument will disable automatic cleanup
#               of the files created by this script.
#
# requires:
#           - bash >= 3.2.57
#           - shellcheck
#
# For each script, it creates a fresh venv, `pip install`s the requirements, and
# finally imports the python script to check for `ImportError`.
#

log() {
    local level=$1 msg=$2
    printf >&2 '%s: %s\n' "$level" "$msg"
}

debug() {
    log DEBUG "$@"
}

info() {
    log INFO "$@"
}

fatal() {
    log FATAL "$@"
    exit 1
}

cleanup() {
    if [[ -n ${workdir+x} && -d $workdir && -w $workdir ]]; then
        info "Removing $workdir"
        local count=0
        rm -rfv -- "$workdir" | while read -r; do
            if (( count++ > 750 )); then
                printf .
                count=0
            fi
        done
        printf '\n'
        info "Removed $workdir"
    fi
}

if [[ ${1-} == nocleanup ]]; then
    shift  # discard nocleanup arg
else
    trap exit INT TERM
    trap cleanup EXIT
fi

this=$(realpath -- "$0"); readonly this
cd "$(dirname "$this")"

shellcheck "$this"

readonly reqs_dir=requirements

if [[ ${1+x} ]]; then
    tmp_dir=$(realpath -- "$1")
    if [[ ! ( -d $tmp_dir && -w $tmp_dir ) ]]; then
        fatal "$tmp_dir is not a writable directory"
    fi
else
    tmp_dir=/tmp
fi

workdir=$(mktemp -d "$tmp_dir/check-requirements.XXXX"); readonly workdir
info "Working directory: $workdir"

check_requirements() {
    local reqs=$1

    info "$reqs: beginning check"
    pip --disable-pip-version-check install -qr "$reqs"
    info "$reqs: OK"
}

check_convert_script() {
    local py=$1             # e.g. ./convert-hf-to-gguf.py
    local pyname=${py##*/}  # e.g. convert-hf-to-gguf.py
    pyname=${pyname%.py}    # e.g. convert-hf-to-gguf

    info "$py: beginning check"

    local reqs="$reqs_dir/requirements-$pyname.txt"
    if [[ ! -r $reqs ]]; then
        fatal "$py missing requirements. Expected: $reqs"
    fi

    local venv="$workdir/$pyname-venv"
    python3 -m venv "$venv"

    (
        # shellcheck source=/dev/null
        source "$venv/bin/activate"

        check_requirements "$reqs"

        python - "$py" "$pyname" <<'EOF'
import sys
from importlib.machinery import SourceFileLoader
py, pyname = sys.argv[1:]
SourceFileLoader(pyname, py).load_module()
EOF
    )

    rm -rf -- "$venv"

    info "$py: imports OK"
}

readonly ignore_eq_eq='check_requirements: ignore "=="'

for req in "$reqs_dir"/*; do
    # Check that all sub-requirements are added to top-level requirements.txt
    if ! grep -qF "$req" requirements.txt; then
        fatal "$req needs to be added to requirements.txt"
    fi

    # Make sure exact release versions aren't being pinned in the requirements
    # Filters out the ignore string
    if grep -vF "$ignore_eq_eq" "$req" | grep -q '=='; then
        tab=$'\t'
        cat >&2 <<EOF
FATAL: Avoid pinning exact package versions. Use '~=' instead.
You can suppress this error by appending the following to the line:
$tab# $ignore_eq_eq
EOF
        exit 1
    fi
done

all_venv="$workdir/all-venv"
python3 -m venv "$all_venv"

(
    # shellcheck source=/dev/null
    source "$all_venv/bin/activate"
    check_requirements requirements.txt
)

rm -rf -- "$all_venv"

check_convert_script convert.py
for py in convert-*.py; do
    check_convert_script "$py"
done

info 'Done! No issues found.'
