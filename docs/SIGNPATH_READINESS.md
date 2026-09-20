# SignPath Foundation readiness dossier

This document is a concise, public description of Yamatana AI IME for code-signing review. It is intended to make project purpose, provenance, runtime privacy and the proposed signing boundary easy to verify from public sources.

## Project identity

- Project: **Yamatana AI IME (MOZC Ver)**
- Repository: https://github.com/YAMA-TANA/fumiori-ai-ime
- Maintainer: https://github.com/YAMA-TANA
- License for Yamatana-authored code: **Apache License 2.0**
- Platform: **Windows 10/11 x64**
- Package type: **WiX/MSI desktop installer**
- Product category: **Japanese Input Method Editor (TSF/Mozc-based) with a local neural reranker**

## What the software does

Yamatana AI IME is a Windows Japanese IME based on Mozc. It preserves Mozc conversion and fallback behavior while optionally reranking conversion candidates with a compact local ONNX model. The reranker can use surrounding text and locally configured document/domain information to choose context-appropriate spellings and homophones.

The AI function is optional and initially disabled. If the AI process is unavailable, times out or declines to override Mozc, normal Mozc candidates remain available.

## Runtime privacy and network behavior

The installed IME performs input processing and AI inference locally. It does **not** transmit the following runtime data to an external AI service or telemetry endpoint:

- typed text
- surrounding/preceding/following text used as context
- conversion candidates
- custom instructions
- dictionary information
- inference results

The application does not contain telemetry, advertising SDKs or automatic crash-report uploading. Runtime behavior is documented in the repository's `PRIVACY.md`.

Development and release tooling may access GitHub, package repositories or model hosting to retrieve public build inputs. Those build-time operations are distinct from installed runtime behavior.

## Public release evidence

Public prereleases exist in GitHub Releases, including:

- `v2.0.0-beta`
- `v2.0.1-beta`

Release assets include the Windows MSI and `SHA256SUMS.txt`. Current beta binaries are explicitly identified as unsigned so users are not led to believe that code signing is already in place.

Releases: https://github.com/YAMA-TANA/fumiori-ai-ime/releases

## Source and dependency provenance

### Mozc

The project builds modified Mozc components from a public fork rather than embedding an opaque prebuilt IME binary.

`build-config.json` currently records:

- fork: `https://github.com/YAMA-TANA/mozc.git`
- pinned fork commit: `45069e109dc1cff2dd55e5b26ef99c848f13ea58`
- corresponding upstream commit: `851c3fe33060d2a6090363e4d7ec44fafde2c03d`

Yamatana-specific renderer/rewriter changes are kept in the public source/build process so that the distributed Mozc derivative can be traced back to source.

### AI model bundle

Large model files are not committed to Git history. `model-manifest.json` pins the public model bundle by URL, expected byte size and SHA-256. The build fetch script accepts the bundle only after digest verification.

Current bundle SHA-256:

`A6F9E21536F6EE30821A0862AF3DE4E0D02DD8CD2FF61D81457B62FC6D7F0615`

The manifest also records per-file byte sizes and SHA-256 digests for the ONNX models, tokenizer, lexical database and model card.

### Third-party notices

Third-party licensing and attribution are documented in:

- `NOTICE`
- `THIRD_PARTY_LICENSES.md`
- the upstream projects referenced by the build configuration and model documentation

Yamatana does not claim authorship of upstream Mozc or third-party model components.

## Build and release provenance

The public repository contains GitHub Actions workflows for CI, Mozc rewriter tests, MSI builds and releases. The release build runs tests and creates the Windows binaries and MSI from public source/configuration.

The intended signing chain is:

1. public source commit/tag
2. GitHub-hosted Windows release workflow
3. unsigned GitHub Actions artifact
4. explicit SignPath signing request using the GitHub artifact ID
5. SignPath-generated signed artifact
6. SHA-256 generation/verification
7. GitHub Release asset

The repository already contains `.github/workflows/signpath.yml`. It is deliberately inactive until real SignPath organization/project/signing-policy identifiers are configured. The SignPath GitHub Action is pinned to a commit SHA rather than a floating version tag.

## Proposed signing boundary

Yamatana signing should cover only product binaries produced by the documented release workflow, principally:

- Yamatana/Mozc-derived IME DLLs produced from the pinned public Mozc fork
- Yamatana AI runtime/tray executables produced from this repository
- supporting Yamatana-built EXE/DLL files required by the installer
- the final Yamatana AI IME MSI

The policy must not sign arbitrary local uploads, pull-request artifacts or unrelated binaries.

Unmodified third-party binaries should not be presented as Yamatana-authored software merely by re-signing them.

## Security and maintenance

The project publishes:

- `SECURITY.md` — vulnerability reporting and release-integrity guidance
- `PRIVACY.md` — runtime privacy behavior
- `CONTRIBUTING.md` — contribution and review expectations
- `CODE_SIGNING_POLICY.md` — signing scope, roles and release controls
- `.github/CODEOWNERS` — review visibility for release/signing/security-sensitive paths

The project is actively maintained and its release/build history is visible through GitHub commits, Actions and Releases.

## Items requiring repository/service settings

The following cannot be completed purely by adding source files and should be enabled/confirmed in GitHub or SignPath settings before or during application:

- branch/ruleset protection for the release branch/workflows
- GitHub Private Vulnerability Reporting
- maintainer MFA
- SignPath Foundation application/approval
- SignPath project, signing policy and Artifact Configuration
- GitHub Actions Variables/Secret containing the identifiers/token issued after adoption

No placeholder SignPath organization/project IDs or credentials are committed to this repository.

## Suggested short application description

> Yamatana AI IME is an Apache-2.0 licensed Windows Japanese Input Method Editor based on a public Mozc fork. It optionally reranks Mozc conversion candidates with a compact ONNX model running entirely on the user's PC; typed text and surrounding context are not sent to cloud AI or telemetry services. Public beta MSI releases, SHA-256 checksums, the modified Mozc source, pinned model hashes, CI/release workflows, privacy policy, security policy and code-signing policy are all available in the public GitHub repositories. We are requesting SignPath Foundation signing so Windows users can verify the publisher and integrity of binaries produced by the public release workflow.
