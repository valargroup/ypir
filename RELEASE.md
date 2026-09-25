# valar-ypir 0.2.1-rc.1 release preparation

This release candidate pins `valar-spiral-rs` to `=0.5.3-rc.1`, matching
published IPIR `0.1.0-rc.3`. This lets voting and Enhance PIR consumers
resolve one compatible Spiral version. No YPIR source or wire-format
changes are included.

## Release order

1. Review and merge this preparation.
2. Validate and publish `valar-ypir 0.2.1-rc.1` from the approved release commit.
3. Verify the new version is available on crates.io.
4. Regenerate the voting preparation lockfile against the registry and run
   its supported checks before publishing `zcash_voting 5.1.1-rc.1`.
5. Update downstream wallet pins and remove the temporary local overrides.

This preparation does not publish crates or create release tags.
