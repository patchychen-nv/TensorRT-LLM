#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --qos=cpu-long
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=trtllm-gateway
#
# The gateway that an agent client is pointed at, run as a chain rather than as
# a single job.
#
# The client holds one URL, and that URL is a scheduler-allocated node. Seven
# days is the longest a job may live here, so the address this deployment
# promises to be stable is guaranteed to move roughly once a week unless
# something puts it back. Two things keep it in place:
#
#   --nodelist pins the successor to the same node, so the IP is unchanged.
#   --dependency=afterany makes it start the moment the predecessor ends,
#     rather than sitting behind whatever else wants a CPU node.
#
# The successor is submitted from inside this job, so the chain continues
# without anybody remembering to extend it. It is submitted at the START, not
# at the end: a job that is killed, preempted or hits its wall clock never runs
# its own cleanup, and those are exactly the cases where the chain must not
# break.
#
# The pin table is shared through --router-state, so the successor inherits
# every conversation's placement and the handover costs no prefix cache. The
# gap is whatever the scheduler takes to start the next job, during which the
# gateway answers nothing and clients retry.
#
# ---------------------------------------------------------------------------
# Gateway generations inside one job
#
# Replacing the gateway binary does not have to cost that gap. A running
# gateway can spawn its own replacement on this node and port -- both bind
# because of SO_REUSEPORT -- hand the listener over and exit. That replacement
# is spawned with start_new_session=True so it outlives its parent, which
# means it is NOT a child of this script and this script cannot wait on it.
#
# This script therefore does not exec the gateway. It stays alive as the job's
# main process and follows a chain of *generations*: it starts generation 0
# itself, and when a generation exits it looks in $GW_DIR/gateway.pid -- which
# every generation rewrites atomically once its own bind has succeeded -- for a
# live successor to follow instead. exec would make the gateway the job's main
# process, so the job would end the moment generation 0 exited and SLURM would
# tear down the successor it had just spawned.
#
# The file is only ever read after the generation being followed has gone, so
# what it says while one is still healthy cannot move this job off it. That
# matters for the aborted handover, where the successor published a pid and was
# then killed: the incumbent is still serving, and the job stays on it whether
# or not anybody gets round to putting the file straight.
#
# Two rules keep that from turning into a job that holds a node forever:
#
#   Adopt only a pid that is alive AND whose /proc cmdline is a gateway.py for
#     THIS --fleet-dir. A pid file is a shared-filesystem artefact that
#     outlives the job that wrote it; a bare number in it proves nothing.
#   Adopt nothing that was already in the file before generation 0 started.
#     That value belongs to a previous job by definition, and is the one way a
#     live-looking stale pid could be inherited.
#
# When neither rule can be satisfied the job ends. That is deliberate: a dead
# gateway releasing the node costs one scheduler gap, and the --dependency
# chain below immediately brings up a fresh one. A job that lingers on a
# maybe-alive pid costs the node and serves nothing.
#
# ---------------------------------------------------------------------------
# Configuration
#
# Nothing site-specific is written into this file. Everything below reads an
# environment variable, and sbatch exports the submitting environment into the
# job by default -- which is also what carries a setting down the chain, since
# each successor is submitted from inside a job that already has it. Set them
# once at the first submit and the whole chain inherits them:
#
#   GW_ROOT=/scratch/.../users/me \
#   GW_ACCOUNT=my_account \
#     sbatch -o /scratch/.../users/me/gw/log/gateway.%j.out gateway.sbatch
#
#   GW_ROOT         required; locates the checkout
#   GW_REPO         this checkout            (default $GW_ROOT/TensorRT-LLM)
#   GW_VAR          what the deployment writes   (default <this dir>/var)
#                   Must be the same directory as fleet.yaml's trace.root:
#                   that is where serve.sh puts the registrations this reads.
#   GW_FLEET_NAME   <cluster_name>_<model.name> from fleet.yaml
#                                            (default kffleet_glm5.2)
#   GW_FLEET_DIR    where backends register  (default $GW_VAR/_fleet/$GW_FLEET_NAME)
#
#   GW_REMOTE_HOST  user@login-node of the cluster the fleet runs on, when that
#                   is not this cluster. Redirects fleetctl, squeue and the
#                   fleet directory together -- see below for why together.
#   GW_REMOTE_DIR   the large_scale_serving directory as that host sees it
#                   serve.sh derives exactly this from trace.root, and a
#                   gateway pointed anywhere else finds no backends and answers
#                   503 to everything while the fleet is perfectly healthy.
#   GW_DIR          users, pins, node file, chain starts   (default $GW_VAR/gw)
#   GW_LOG_DIR      successor --output       (default $GW_DIR/log)
#   GW_SELF         script the chain resubmits   (default this file in GW_REPO)
#   GW_PORT         listen port              (default 8333)
#   GW_FLEET_CONFIG fleet.yaml for preemption recovery (default: off)
#   GW_RELAY_ARGS   replaces the default --no-relay. For walltime relay:
#                   "--yaml <fleet.yaml> --relay-per-instance --lead-time 3300"
#                   (lead must exceed the ~50min cold start)
#   GW_EXTRA_POOLS  more fleet directories behind the same address,
#                   whitespace-separated, each a gateway.py --pool spec:
#                     NAME=FLEET_DIR[,FLEET_CONFIG]
#                   e.g. "kimi-k3=$GW_VAR/_fleet/kffleet_kimi-k3,<fleet_k3.yaml>".
#                   Every directory's backends are one routing pool: requests,
#                   retries and conversation pins treat them alike, whatever
#                   model a request names. FLEET_CONFIG drives revive, recovery
#                   and -- with --relay-per-instance in GW_RELAY_ARGS --
#                   walltime relay for that pool's jobs alone; without one the
#                   pool is routed but never supervised. A chain already
#                   running never sees a change here (every link inherits the
#                   environment it was submitted with), so to add a pool to a
#                   live gateway put the same spec in $GW_DIR/pools.conf, one
#                   per line, and POST /_gateway/handover: every generation
#                   reads that file when it starts.
#   GW_MAX_STARTS   chain-thrash threshold   (default 5 in 30 minutes)
#   GW_EXTRA_ARGS   appended to the gateway command line
#   GW_NO_CHAIN     set to anything to run once without a successor
#
#   GW_PIDFILE_WAIT  seconds generation 0 gets to publish its pid
#                                            (default 60)
#   GW_HANDOVER_GRACE seconds to look for a successor generation after one
#                    exits, before ending the job        (default 60)
#   GW_SHUTDOWN_GRACE seconds a signalled generation gets to drain before it
#                    is killed outright                  (default 150). This is
#                    a ceiling, not a promise: on a cancelled or timed-out job
#                    SLURM's own KillWait sends SIGKILL first, so set it below
#                    KillWait for the drain to be the thing that decides.
#
# The scheduler options are separate, because #SBATCH directives are read
# before this shell runs and cannot see a variable. SLURM resolves them as
# command line > environment > directive, and the SBATCH_* environment
# variables propagate down the chain the same way the GW_* ones do:
#
#   GW_ACCOUNT -> SBATCH_ACCOUNT, or export SBATCH_PARTITION / SBATCH_QOS /
#   SBATCH_TIMELIMIT / SBATCH_JOB_NAME directly. There is no SBATCH_OUTPUT, so
#   the first job's log path is an `sbatch -o` argument; successors get
#   GW_LOG_DIR, which is why that one is a variable rather than a directive.
set -uo pipefail

if [ -z "${GW_ROOT:-}" ]; then
    echo "GW_ROOT is not set. It is the one path everything else derives from;" >&2
    echo "see the configuration block at the top of this file." >&2
    exit 2
fi

REPO="${GW_REPO:-$GW_ROOT/TensorRT-LLM}"
HERE="$REPO/examples/serve/large_scale_serving"
# Defaults sit under the checkout because that is where fleet.yaml's trace.root
# points: the gateway and the serving jobs have to meet in one directory, and
# deriving both from the deployment rather than from two unrelated roots is
# what stops them drifting apart.
VAR="${GW_VAR:-$HERE/var}"
FLEET_DIR="${GW_FLEET_DIR:-$VAR/_fleet/${GW_FLEET_NAME:-kffleet_glm5.2}}"
GW_OWN="${GW_DIR:-$VAR/gw}"
LOG_DIR="${GW_LOG_DIR:-$GW_OWN/log}"
PORT="${GW_PORT:-8333}"
MAX_STARTS="${GW_MAX_STARTS:-5}"
# Not $0: SLURM runs a spool copy of this script, so $0 is a path that exists
# only for the life of the job and cannot be resubmitted.
SELF="${GW_SELF:-$HERE/gateway.sbatch}"

# Exported so the sbatch that queues the successor inherits it, rather than
# each link in the chain having to be told again.
[ -n "${GW_ACCOUNT:-}" ] && export SBATCH_ACCOUNT="$GW_ACCOUNT"

NODE=$(hostname -s)
mkdir -p "$GW_OWN" "$LOG_DIR" "$FLEET_DIR"

echo "GATEWAY_NODE=$NODE"
echo "GATEWAY_URL=http://$NODE:$PORT"
echo "$NODE" > "$GW_OWN/node"

for path in "$SELF" "$HERE/gateway.py"; do
    if [ ! -f "$path" ]; then
        echo "WARNING: $path does not exist; check GW_REPO/GW_SELF" >&2
    fi
done

# A chain that submits its successor at the start will, if the gateway dies
# immediately, submit a successor that also dies immediately -- a submission
# loop against a shared scheduler. So record every start and refuse to extend
# the chain when it is clearly thrashing. Refusing costs the chain; continuing
# costs everyone else on the cluster, which is worse.
STARTS="$GW_OWN/chain_starts"
date +%s >> "$STARTS"
RECENT=$(awk -v cutoff="$(( $(date +%s) - 1800 ))" '$1 > cutoff' "$STARTS" 2>/dev/null | wc -l)
tail -200 "$STARTS" > "$STARTS.tmp" 2>/dev/null && mv -f "$STARTS.tmp" "$STARTS"

if [ "$RECENT" -gt "$MAX_STARTS" ]; then
    echo "REFUSING to queue a successor: $RECENT chain starts in the last 30 min."
    echo "The gateway is failing fast and the chain would spin. Fix it, then"
    echo "restart the chain by hand with: sbatch --nodelist=$NODE $SELF"
elif [ -z "${GW_NO_CHAIN:-}" ]; then
    # --output is passed rather than left to the directive: there is no
    # SBATCH_OUTPUT, so this is the only way a successor's log follows the
    # configured directory instead of wherever the first submit happened to
    # put it.
    NEXT=$(sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID" \
        --nodelist="$NODE" \
        --output="$LOG_DIR/gateway.%j.out" \
        "$SELF" 2>&1) && echo "successor queued: $NEXT on $NODE" \
        || echo "WARNING: could not queue a successor: $NEXT"
fi

# The predecessor may still be releasing the port. Wait rather than exit: this
# job already holds the node, so there is nothing to fall back to and failing
# here would break the chain it was started to continue.
for attempt in $(seq 1 60); do
    if ! ss -lnt 2>/dev/null | grep -q ":$PORT "; then
        break
    fi
    [ "$attempt" = 1 ] && echo "waiting for :$PORT to be released by the predecessor"
    sleep 5
done

# A gateway serving a fleet on a *different* cluster needs three things
# redirected, and they have to move together. GW_REMOTE_HOST is one switch for
# all three rather than three knobs, because any subset of them is wrong in a
# way that looks fine: a local fleetctl on the gateway's own cluster execs
# happily and reconciles the wrong scheduler, and a local squeue answers
# "cannot tell" forever.
REMOTE_ARGS=()
if [ -n "${GW_REMOTE_HOST:-}" ]; then
    : "${GW_REMOTE_DIR:?set it to the large_scale_serving directory on the fleet cluster}"
    export FLEETCTL_SSH_HOST="$GW_REMOTE_HOST"
    export FLEETCTL_REMOTE_DIR="$GW_REMOTE_DIR"
    export SLURM_SSH_HOST="$GW_REMOTE_HOST"
    REMOTE_ARGS=(--fleetctl "$HERE/fleetctl-remote" --slurm-wrapper "$HERE/slurm-remote")

    # Backends register by dropping a file into the fleet directory, and that
    # directory is on the other cluster's filesystem. Pull it; the gateway only
    # ever reads it. (If the fleet's compute nodes can reach this host over
    # HTTP, gateway.register_url is simpler and this is unnecessary -- but that
    # depends on the gateway's network, so it is not assumed here.)
    export FLEET_SYNC_SOURCE="$GW_REMOTE_HOST:$GW_REMOTE_DIR/var/_fleet/${GW_FLEET_NAME:-kffleet_glm5.2}/"
    export FLEET_SYNC_DEST="$FLEET_DIR"
    "$HERE/fleet-sync" >> "$LOG_DIR/fleet-sync.log" 2>&1 &
    echo "fleet-sync started (pid $!) pulling $FLEET_SYNC_SOURCE"
fi

# Preemption recovery is off unless a fleet.yaml is named: it is the only
# option here that lets the gateway submit jobs, so it is opt-in rather than
# something a bare GW_ROOT switches on.
RECOVERY=()
if [ -n "${GW_FLEET_CONFIG:-}" ]; then
    RECOVERY=(--fleet-config "$GW_FLEET_CONFIG")
    echo "preemption recovery enabled against $GW_FLEET_CONFIG"
    if [ -n "${GW_REMOTE_HOST:-}" ]; then
        echo "  ... on $GW_REMOTE_HOST, via fleetctl-remote and slurm-remote"
    fi
fi

# Extra pools (fleet directories). read -a rather than an unquoted expansion, so a spec is
# split on whitespace and nothing else -- never globbed against this directory.
POOLS=()
read -r -a POOL_SPECS <<< "${GW_EXTRA_POOLS:-}"
for spec in "${POOL_SPECS[@]}"; do
    POOLS+=(--pool "$spec")
    echo "extra pool: $spec"
done
if [ "${#POOL_SPECS[@]}" -gt 0 ] && [ -n "${GW_REMOTE_HOST:-}" ]; then
    echo "WARNING: fleet-sync pulls only the default pool's directory; an extra pool's" >&2
    echo "WARNING: FLEET_DIR has to be kept in sync from $GW_REMOTE_HOST some other way." >&2
fi
if [ -f "$GW_OWN/pools.conf" ]; then
    echo "pools file: $GW_OWN/pools.conf (read by every gateway generation at startup)"
fi

echo "fleet dir: $FLEET_DIR"

# ---------------------------------------------------------------------------
# Generation tracking
#
# Everything below exists to answer one question repeatedly: is there still a
# gateway on this node that this job is responsible for? While the answer is
# yes the job stays up; the first time it is no, the job ends.
PIDFILE="$GW_OWN/gateway.pid"
PIDFILE_WAIT="${GW_PIDFILE_WAIT:-60}"
HANDOVER_GRACE="${GW_HANDOVER_GRACE:-60}"
SHUTDOWN_GRACE="${GW_SHUTDOWN_GRACE:-150}"

# The pid the gateway publishes, or nothing. Never the raw file contents: an
# unwritten, truncated or hand-edited file must read as "no pid" rather than
# as a word that later gets passed to kill.
read_pidfile() {
    local pid=""
    [ -r "$PIDFILE" ] || return 0
    pid=$(head -n 1 "$PIDFILE" 2>/dev/null | tr -d '[:space:]')
    case "$pid" in
        '' | *[!0-9]*) return 0 ;;
    esac
    printf '%s\n' "$pid"
}

# Is this pid a gateway of ours, right now? This, never a bare `kill -0`, is
# what "still running" means here, for two reasons.
#
# Pids are recycled, and $PIDFILE lives on a shared filesystem that outlives the
# job that wrote it, so "the number in the file is running" is satisfied by any
# unrelated process that happened to land on it. The cmdline is checked for
# gateway.py and for this deployment's --fleet-dir, which together no other
# process on the node has a reason to carry.
#
# And a zombie answers `kill -0` successfully for as long as nobody reaps it --
# but its cmdline is empty, so it fails here. That is the difference between
# ending a job whose gateway is gone and holding a node on a corpse, and a
# generation we adopted is exactly the process we cannot reap ourselves.
#
# Unreadable cmdline counts as no: this whole path is about refusing to hold a
# node on a guess.
is_gateway() {
    local cmd=""
    kill -0 "$1" 2>/dev/null || return 1
    [ -r "/proc/$1/cmdline" ] || return 1
    cmd=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null) || return 1
    case "$cmd" in *gateway.py*) ;; *) return 1 ;; esac
    case "$cmd" in *"$FLEET_DIR"*) ;; *) return 1 ;; esac
    return 0
}

# Whatever a previous job left behind. A successor is only ever adopted if the
# file has moved on from this, which is what stops a stale pid -- the one
# failure that both looks alive and belongs to nobody -- from being followed.
PRIOR=$(read_pidfile)
if [ -n "$PRIOR" ]; then
    echo "note: $PIDFILE already names pid $PRIOR; treating it as stale"
fi

# Forward the signal and get out of the way. SLURM asking for the node back is
# not a handover: no successor is adopted after this point, the current
# generation is given SHUTDOWN_GRACE to finish what it is holding, and then it
# is killed so the drain cannot outlast the job's own grace period.
SIGNALLED=""
GEN_PID=""
WATCHDOG=""
on_signal() {
    local target="$GEN_PID"
    SIGNALLED="$1"
    echo "received SIG$1; forwarding to gateway generation pid $target"
    kill -"$1" "$target" 2>/dev/null
    # The watchdog exists so that a generation which will not finish draining
    # cannot keep the allocation past the grace SLURM itself allows. It rechecks
    # identity first: by the time it fires the pid may have been recycled, and a
    # timer that SIGKILLs whatever now holds that number is a worse bug than the
    # one it is guarding against.
    (
        sleep "$SHUTDOWN_GRACE"
        if is_gateway "$target"; then
            echo "generation $target did not drain in ${SHUTDOWN_GRACE}s; killing it"
            kill -KILL "$target" 2>/dev/null
        fi
    ) &
    WATCHDOG=$!
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP

# Not exec: see the generation note at the top of this file. $! is the python
# process itself rather than a wrapper shell, so it is the same pid the gateway
# publishes, and the startup check below can compare the two.
python3 "$HERE/gateway.py" \
    --fleet-dir "$FLEET_DIR" \
    --users "$GW_OWN/users.txt" \
    --host 0.0.0.0 --port "$PORT" \
    --router-state "$GW_OWN/router_state.json" \
    "${RECOVERY[@]}" \
    "${REMOTE_ARGS[@]}" \
    "${POOLS[@]}" \
    ${GW_RELAY_ARGS:---no-relay} --log-level INFO \
    ${GW_EXTRA_ARGS:-} &
GEN_PID=$!
GEN_OWNED=1
echo "gateway generation 0 started (pid $GEN_PID)"

# Wait for generation 0 to publish itself. This is diagnostic, not a gate --
# the job runs fine either way -- but it is the only place the missing half of
# the contract shows up as a message instead of as a handover that silently
# takes the node down with it. Seeing our own pid in the file also retires
# PRIOR: the file is demonstrably being maintained by this job, so a stale
# value can no longer be sitting in it.
PIDFILE_LIVE=0
for attempt in $(seq 1 "$PIDFILE_WAIT"); do
    if [ "$(read_pidfile)" = "$GEN_PID" ]; then
        PIDFILE_LIVE=1
        break
    fi
    kill -0 "$GEN_PID" 2>/dev/null || break
    sleep 1
done
if [ "$PIDFILE_LIVE" = 1 ]; then
    echo "gateway publishes its pid to $PIDFILE; handovers will be followed"
    PRIOR=""
else
    echo "WARNING: $PIDFILE does not name pid $GEN_PID after ${PIDFILE_WAIT}s." >&2
    echo "WARNING: this gateway does not publish its pid, so a hot handover" >&2
    echo "WARNING: cannot be followed and would be killed with this job." >&2
fi

# Look for a generation that replaced the one that just went away. Three
# conditions, and all of them are about refusing rather than accepting: not the
# one that just died, not what a previous job left in the file, and actually a
# gateway. The grace period is for the filesystem, not for the gateway -- the
# successor publishes its pid at startup, long before its predecessor exits, so
# in a real handover the answer is already there and only a shared-filesystem
# attribute cache can delay it.
find_successor() {
    local deadline=$(( $(date +%s) + HANDOVER_GRACE )) pid=""
    while [ -z "$SIGNALLED" ]; do
        pid=$(read_pidfile)
        if [ -n "$pid" ] && [ "$pid" != "$GEN_PID" ] && [ "$pid" != "$PRIOR" ] \
            && is_gateway "$pid"; then
            printf '%s\n' "$pid"
            return 0
        fi
        [ "$(date +%s)" -ge "$deadline" ] && return 1
        sleep 2
    done
    return 1
}

# Give the current generation until the deadline to go away on its own. Used on
# the way out, where the only thing left to decide is whether to wait or not.
await_exit() {
    local deadline=$(( $(date +%s) + $1 ))
    while is_gateway "$GEN_PID" && [ "$(date +%s)" -lt "$deadline" ]; do
        sleep 1
    done
}

RC=0
while : ; do
    if [ "$GEN_OWNED" = 1 ]; then
        # Our own child, so wait gives both the moment and the reason it left.
        # A trapped signal makes wait return early with >128 before the child
        # has actually gone; go round again until it really has, which the
        # watchdog in on_signal guarantees is bounded.
        wait "$GEN_PID"
        RC=$?
        while [ "$RC" -gt 128 ] && kill -0 "$GEN_PID" 2>/dev/null; do
            wait "$GEN_PID"
            gw_again=$?
            [ "$gw_again" = 127 ] && break
            RC=$gw_again
        done
        echo "gateway generation (pid $GEN_PID) exited rc=$RC"
    else
        # An adopted generation is not our child, so there is no status to
        # collect and nothing to reap -- only the question of whether it is
        # still there. Polled rather than waited on, and polled often: the
        # interval is dead time at the front of the successor search, and a
        # generation that hands over and exits inside it would be missed.
        # Confirmed once before acting on a negative, so that a single bad
        # /proc read cannot end a job whose gateway is fine.
        while : ; do
            [ -n "$SIGNALLED" ] && break
            if ! is_gateway "$GEN_PID"; then
                sleep 1
                is_gateway "$GEN_PID" || break
            fi
            sleep 2
        done
        if [ -z "$SIGNALLED" ]; then
            RC=0
            echo "adopted gateway generation (pid $GEN_PID) is gone"
        fi
    fi

    if [ -n "$SIGNALLED" ]; then
        await_exit "$SHUTDOWN_GRACE"
        echo "ending the job after SIG$SIGNALLED"
        break
    fi

    # find_successor runs in a command substitution, and bash resets traps in
    # one -- so a signal arriving during the search is seen by this shell but
    # not by that subshell. Re-test rather than act on its answer.
    NEXT=$(find_successor)
    if [ -n "$SIGNALLED" ]; then
        await_exit "$SHUTDOWN_GRACE"
        echo "ending the job after SIG$SIGNALLED"
        break
    fi
    if [ -z "$NEXT" ]; then
        echo "no gateway generation remains; ending the job so the node is"
        echo "released and the queued successor can start a fresh one"
        break
    fi
    echo "following gateway generation pid $NEXT (handover from $GEN_PID)"
    GEN_PID="$NEXT"
    GEN_OWNED=0
done

# Nothing may outlive this script. What happens to a generation still running
# when the job ends is a SLURM configuration detail: proctrack/cgroup kills it
# with the job step, proctrack/linuxproc walks the process tree and a successor
# in its own session has already walked out of that tree -- in which case it
# survives as an orphan still holding :$PORT on a node the scheduler has by then
# given to somebody else. Neither outcome is one to leave to configuration. By
# here the only generation that can still be alive is one that outlasted the
# shutdown grace, so end it explicitly.
if is_gateway "$GEN_PID"; then
    echo "generation $GEN_PID outlived the shutdown grace; killing it so the job"
    echo "does not leave an orphan holding :$PORT"
    kill -KILL "$GEN_PID" 2>/dev/null
fi
# A watchdog left sleeping would outlive this shell and SIGKILL whatever ends up
# holding that pid. It has a job only while the shell is still here to be held
# up by a slow drain.
[ -n "$WATCHDOG" ] && kill "$WATCHDOG" 2>/dev/null
exit "$RC"
