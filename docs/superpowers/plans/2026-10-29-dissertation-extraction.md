# Dissertation companion extraction implementation plan

Goal: implement the user-approved chapter-based extraction in the existing destination checkout.

Architecture: chapter-local scripts and retained artifacts, one shared Python utility tree, independent data paths, and explicit source provenance. Preserve the source checkout and existing dissertation edits.

- [x] Extract current artifacts and original CNN evidence with checksums.
- [x] Adapt imports and filesystem paths, retain relevant regression checks, verify standalone analyses.
- [x] Write readmaker README with the current dissertation abstract, chapter guides, licenses and citation metadata.
- [x] Update Appendix C with scribe revision markup and compile.
- [x] Verify artifact integrity, licensing notices, documentation links and repository diffs.

User-approved defaults: Apache 2.0 for author-owned code, CC BY 4.0 for original prose and figures, selected final models only, no datasets or dissertation submodule. No commits, branches or remote publication.
