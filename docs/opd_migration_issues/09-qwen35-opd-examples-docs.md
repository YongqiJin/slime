# Issue: Add Qwen3.5 OPD Production And Smoke Docs

## Goal

Document the final Qwen3.5-27B multi-domain OPD workflow and the two-node smoke workflow.

## Scope

- Update `examples/on_policy_distillation/qwen3_5_multidomain/`.
- Document production Qwen3.5-27B student with multiple Qwen3.5-27B domain teachers.
- Document two-node Qwen3.5-9B/4B smoke.
- Show teacher pool start, training launch, and teacher pool stop commands.
- Keep scripts as short entry points.

## Non-Goals

- Do not hard-code personal checkpoint paths.
- Do not migrate the original Uni-OPD `zz/opd` stack verbatim.
- Do not make large Qwen3.5 runs mandatory CI.

## Acceptance Criteria

- README describes required env vars and expected files.
- Smoke command path uses the same teacher lifecycle as production.
- Docs state that current Qwen3.5 model support comes from slime, not Uni-OPD qwen3-next compatibility edits.

## Dependencies

- Issue 03.
- Issue 04.
