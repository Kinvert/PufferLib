# G240 receipt audit — September 15, 2026

Received archive: `eval-profile-handoff.wGPixsmA.tar.gz`, SHA256 `b1ea1074374e0f7a5dabd37e5b16787209b84a7c65225548160e3a7f936a65ae`.

Validated safe relative paths, regular files/directories only, unique file paths and full `PACKAGE.sha256` coverage before extraction. **764 manifest entries passed**, plus the manifest itself: 765 files, 12,176,567 unpacked bytes. Original contents/names are unchanged; this receipt and the scoped `.gitignore` are local additions.

Independently checked all **57 attempts** against raw output-log hashes, exit-code and wall-time files: nine validation, 32 panel and 16 performance attempts. Parsed the raw final CUDA_EVAL records: **12 successful old/new pairs match exactly** (four validation plus eight controls), and all **eight original-timeout checkpoint pairs timed out in both diagnostic arms**. All **90 included captured source files** match `audit/source.sha256`; two Python bytecode-cache files listed in that broader manifest were intentionally absent from the package. These are not missing native source files.

The recorded checkpoint hashes, finiteness/repeatability checks, 1,927 unchanged original-file identities and full Nsight trace checks were performed remotely. Actual checkpoint/binary files and Nsight trace databases are absent here, so those deeper checks were **not independently repeated on G240**. Package integrity does not by itself establish every derived statistical/profiling claim. The original-context directory contains summary data/protocols, not the complete original 320-evaluation raw archive.

Read [remote findings](audit/FINDINGS.md), [profiling coverage](audit/PROFILE.md), [raw paired results](audit/REPORT.md), and the [proposed evaluation v2](audit/PONG_EVALUATION_POLICY_PROPOSAL.md). The proposal is not implemented or authorized by receipt of this archive. Archived scripts and patches were inspected as evidence; none were executed or applied to the working source during intake.

Main conclusions supported by the diagnostic receipts: sparse match completion despite advancing simulation; no consistent observed telemetry speedup; original missing outcomes remain missing. Summed traced GPU kernel durations are not elapsed process times. Shared GEMM attribution remains incomplete; original CNN architecture quality claims are unchanged.
