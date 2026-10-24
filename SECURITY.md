# Security policy

## Supported versions

Only the latest commit on `main` receives fixes. There are no maintained release branches.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's advisory form:
https://github.com/hellpuffyt/dataset-profiler/security/advisories/new

Do not open a public issue for a suspected vulnerability. Include the version or commit,
a minimal reproduction, and the impact you observed. You should receive an acknowledgement
within a few days; fixes are released as soon as they are verified.

## Scope

dataset-profiler reads CSV files and JSON baseline profiles and writes reports. It performs no network I/O and never executes input content. Relevant reports include: crafted input that causes unbounded memory despite the documented limits, a baseline file that causes code execution or writes outside `--output`, or a drift result that is silently wrong for valid input.
