# Security Policy

## Supported Versions

Security fixes are handled on the default branch.

## Reporting a Vulnerability

Please report security-sensitive issues privately to the repository owner through GitHub. Do not open a public issue for vulnerabilities that could expose credentials, private datasets, or remote execution risks.

When reporting, include:

- affected commit or version
- reproduction steps
- impact assessment
- any relevant logs or configuration files, with secrets removed

## Data and Credential Safety

- Do not commit private datasets, access tokens, SSH keys, cloud credentials, or service logs containing sensitive information.
- Keep run artifacts under `outputs/`; this directory is ignored by Git.
- Review logs before sharing them publicly.
