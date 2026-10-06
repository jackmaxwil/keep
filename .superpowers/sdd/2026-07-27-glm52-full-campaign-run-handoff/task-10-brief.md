# Task 10 brief — mount-free worker task, reboot-safe units, and guarded CLI

## Position in the implementation

Task 10 starts only after Task 9 is independently approved. It implements
candidate-thirteen slice 10 and the user's explicit missing production-route
gate. It must produce the genuine guarded production entrypoint, mount-free
Sky task, worker bootstrap/systemd/deadline contracts, and authenticated
graceful-stop evidence. It must not call live AWS/Sky, deploy, launch, or
claim the later measured decision closure.

Read first:

- candidate-thirteen architecture sections 8 and 9, implementation slice 10,
  SkyPilot compatibility tests, deadline and worker-drain sections;
- the freeze addendum and full-run handoff Phase One sections 5.4 and 5.5;
- approved Tasks 7–9 interfaces/reports;
- current `submit_sky_campaign.py`, `submit_sky_campaign.sh`, campaign
  descriptor/task/YAML builders, bootstrap, systemd, deadline, run, and
  break-glass code under `aws/glm52-gpu/`;
- existing Sky campaign/submission/fence/terminal tests.

## Constraints

- Work only in `/Users/jack.mazac/Developer/keep`; preserve concurrent work.
- Strict named RED-to-GREEN TDD with literal transcripts.
- No Git mutation, AWS/network call, Sky POST, deployment, launch,
  termination, or billing effect.
- Do not modify the installed SkyPilot environment.
- Enforcement remains Python 3.9 import-light; production glue remains
  dependency-injectable with no hidden retries.
- Do not add a marker string or weaken/suppress the production-route
  assertion. Implement the actual route it requires.
- Production can be reached only through the guarded action-consumed path.
  Raw `sky jobs launch`, the public retrying client, arbitrary YAML, caller
  command text, and legacy EC2 launch scripts remain unreachable.

## Mount-free immutable Sky task

Render one canonical task whose name exactly equals the accepted claim/H.1c
job name and which fixes:

- one node, AWS `us-west-2`, on-demand `p5.48xlarge`,
  `use_spot: false`, max `$55.04/hour`;
- bounded recovery with application retry disabled;
- `api_server_access: false`;
- no local `file_mounts`, no local `workdir`, and no pre-POST mount upload or
  subprocess;
- only immutable descriptor URI/SHA, approval URI/SHA, intent URI/hashes,
  managed mode, expected job name, and exact archive authority as initial
  inputs;
- setup that downloads/authenticates the descriptor and one prebuilt
  repository tar, extracts it to `/opt/keep-campaign/repo`, and invokes only
  the tar-pinned bootstrap.

Freeze the exact parsed SkyPilot `0.13.0` YAML and exact `JobsLaunchBody`
fields:

```text
env_vars
entrypoint
entrypoint_command
using_remote_api_server
override_skypilot_config
override_skypilot_config_path
file_mounts_blob_id
client_api_version
task
name
pool
num_jobs
```

Bind the exact endpoint `/jobs/launch`, API/version headers, bearer format,
closed task/body/YAML identities, service-account user, job name, request
tuple, and Task 9 admission envelope. No alternate field, nullability,
endpoint, raw HTTP retry, or caller command/path is accepted.

## Worker bootstrap and exact units

The authenticated tar installs exactly:

```text
keep-glm52-campaign.service
keep-glm52-deadline.service
keep-glm52-deadline.timer
```

Freeze and validate:

- campaign `Restart=no`, `KillMode=control-group`,
  `TimeoutStopSec=1200`, `SendSIGKILL=no`;
- fixed working directory `/opt/keep-campaign/repo`;
- fixed `ExecStart` invoking only
  `/opt/keep-campaign/repo/aws/glm52-gpu/scripts/run_campaign.sh` with the
  authenticated descriptor path and deadline;
- fixed `ExecStartPre` deadline guard;
- heavy-job lock, `/mnt/nvme/glm52-campaign`, exact resume directories,
  `/etc/keep-glm52/campaign.json`, and
  `/var/lib/keep-glm52/deadline-state.json`;
- no caller-selectable shell, path, unit, signal, command, service, or
  environment authority.

Bootstrap must write and hash the units and signed deadline state, run the
fixed daemon-reload/enable/start/readback sequence, then publish marker-last
`BOOTSTRAP_READY.json` and append the hash-linked BOOTSTRAP ledger record.
Both bind unit hashes, signed deadline state, enabled/active/timer/process
readback, descriptor/deadline, instance/allocation, archive, resume
directories, and exact S3 VersionIds. Any mismatch prevents phase advance.

## Reboot-safe deadline behavior

The timer uses immutable deadline-derived `OnCalendar` drop-ins and
`Persistent=true`. `/run` is never the sole deadline memory. On boot, timer
delivery, and `ExecStartPre`, authenticate descriptor plus persistent state
and derive:

- before `T-60m`: start permitted and no stop marker;
- at/after `T-60m`: atomically recreate `/run/keep-glm52/STOP`, persist the
  edge, forbid new chunks/windows;
- at/after `T-50m`: recreate/preserve stop, refuse new start, run the fixed
  checkpoint/sync stop path;
- at/after `T-30m`: never restart or force-kill; only the retained,
  controller-quiesced Task 12 force authority can terminate remnants.

At `T-50m`, only `systemctl stop keep-glm52-campaign.service` is permitted.
Systemd sends SIGTERM to the full control group. `SendSIGKILL=no` means a
survivor after 1,200 seconds remains visible for retained force handling.

Implement deterministic simulation/readback interfaces for reboot before and
after `T-60m` and `T-50m`, restart attempts at every edge, wall-clock
reconstruction, stale/missing/corrupt persistent state, child survival beyond
timeout, checkpoint sync, and exact systemd result fields.

## Parameterless worker drain and evidence

Render the exact parameterless `KeepGlm52GracefulStopV1` SSM document. Only
the retained worker-drain-signal role may invoke it and only on exact
activation-tagged workers. It has no command/path/unit/signal/shell parameter.
Every other role is denied `ssm:SendCommand`; every role is denied SSM
sessions.

Authenticate `WORKER_GRACEFUL_STOP.json` with exact unit/script hashes,
`ActiveState`, `SubState`, `Result`, `ExecMainCode`, `ExecMainStatus`,
control-group emptiness, stop-file identity, checkpoint/latest, campaign
terminal marker, and SSM command identity or exact null for local-timer
authority. Missing checkpoint/service evidence never becomes successful
drain.

## Genuine guarded production CLI

Extend the existing guarded submitter with a real `production` mode:

- shell and Python entrypoints accept the exact production invocation;
- require canonical authenticated production descriptor, archive, approvals,
  Task 8 authority/spend reserve, Task 9 launch/admission/custody, and
  `H100_RESUME_READY.json`;
- require action-consumed current activation/generation and exact one-wire
  body;
- qualification, cache-seed, and production remain distinct immutable modes
  under one cumulative spend envelope;
- validation/dry-run cannot POST or reserve;
- execution accepts only the exact injected Task 9 admission/relay boundary,
  never raw Sky/HTTP/subprocess/EC2;
- one action permits at most one production submission;
- unresolved, stale, duplicate, foreign, missing, expired, or already
  consumed authority fails before POST;
- output is nonsecret and canonical; ambiguous send is handed to Task 9
  readback/liability custody without a second submission.

The focused suite must pass because the executable production route exists,
not because an assertion is removed, skipped, xfailed, marker-matched, or
weakened.

## Required named RED-to-GREEN coverage

At minimum:

1. production mode absent RED followed by genuine guarded route GREEN;
2. exact mount-free parsed Sky `0.13.0` YAML and closed body fields;
3. no local mount/workdir/upload/subprocess before POST;
4. exact endpoint/headers/token/user/job/task/envelope identities;
5. wrong/extra/missing body/YAML/env/nullability mutants;
6. archive/descriptor/approval/intent/H100-resume/action/spend/custody
   identity mutants;
7. validation/dry-run zero reserve and zero POST;
8. one action/one POST under success, rejection, timeout, connection loss,
   replay, and process death;
9. raw Sky/public retrying client/legacy EC2 launch unreachability;
10. exact three unit files and byte/hash/readback identities;
11. bootstrap marker-last plus ledger ordering and failed-start/missing-timer
    rejection;
12. exact `Restart`, kill mode, timeout, no-SIGKILL, paths, commands, and
    heavy-lock/resume state;
13. persistent signed deadline reconstruction across every reboot edge;
14. `T-60m`, `T-50m`, `T-30m` start/stop/assignment behavior;
15. child survival beyond timeout with no systemd SIGKILL;
16. exact parameterless SSM document/role/tag and all session/parameter
    denials;
17. graceful-stop record and checkpoint/systemd/control-group mutants;
18. qualification/cache-seed/production separation and shared spend;
19. shell syntax/executable mode/canonical no-overwrite outputs;
20. Python 3.9/3.12 import, focused/aggregate/CloudFormation compatibility,
    Ruff, and `git diff --check`.

## Verification and report

Run focused Task 10 plus existing Sky campaign, submission integration,
production fence/audit, terminal state, enforcement aggregate/migration, and
frozen CloudFormation tests. Run Python 3.9 compilation/import-light,
Python 3.12 production-adapter import, exact Sky parser compatibility, Ruff
`E4,E7,E9,F`, `bash -n`/`zsh -n` as applicable, executable-mode and canonical
artifact checks, and `git diff --check`.

Write
`.superpowers/sdd/2026-07-27-glm52-full-campaign-run-handoff/task-10-report.md`
with owned/integration files, literal RED/GREEN, exact commands/counts,
hashes, self-review, and limitations. State explicitly that all POST/systemd/
SSM/AWS effects use injected simulators and are not live or deployed truth.

Return only status, owned-file summary, one-line verification counts, and
concerns. Leave changes unstaged.
