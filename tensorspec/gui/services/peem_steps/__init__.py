"""XMCD-imaging steps of the PEEM service, one mixin per workflow step.

XMCD-imaging service of the PEEM suite: wraps the PEEM service and adds the UE49 / UView / P3B
dichroism and domain workflow (no Qt).

``PeemService`` (gui/services/peem_service.py) is the one PEEM service; it inherits these mixins
(load, cleanup, drift_stack, image_analysis, maps_domains; shared helpers in common.py) next to
its original stack / pair / drift / BG / sum-rule methods. The XMCD-imaging methods are:

    load_xpeem       UE49 scan tables, UView folders (BNL ESM) or .P3B folders (ALS)
                     -> one PEEM dataset (format detected; core/io/xpeem_loaders.py)
    load_ue49 / load_uview  the same for one format (kept for scripts)
    drift_correct    /processed           step 1: every frame aligned to frame 0 (drift-corrected stack)
    stack            /analysis/stack      step 2: one image per polarization (mean), B aligned onto A
    asymmetry        /analysis/pcd|pld    step 3: (A-B)/(A+B) and the plain difference
    compute_pcd      /analysis/pcd        steps 1-3 in one call
                     (sign defaults to the loaded beamline's convention)
    compute_pld      /analysis/pld        linear dichroism asymmetry (one energy)
    compute_pmld     /analysis/pld, pmld, pmnld from LH/LV energy stacks
    roi_statistics   ROI mean, background ROI, f+/f-/f0 on any map
    compute_domain_map  /analysis/domain_map from pcd (or pmld)

/raw is not written after loading: stack clean-up edits live in /analysis/stack_edits, and every
step above appends a "PEEM <step>: {parameters}" line to /history.
"""
