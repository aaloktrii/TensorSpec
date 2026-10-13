"""PEEM back end: the calls any front end uses (Qt panels, scripts, a web page).

``PeemService`` (gui/services/peem_service.py) is the one PEEM service. It holds no front-end state:
datasets, regions, profiles and results live in the workspace (``/raw``, ``/processed``,
``/analysis/<kind>``, ``/history``), it imports no Qt, and its calls take and return plain Python /
numpy values. A front end only draws what these calls return and passes on what the user clicks or
types. The math is in core/ (core/io, core/peem); drawing helpers in plotting/peem_plots.py.

Contract, by workflow step (module in peem_steps/ in brackets; tests/test_peem_backend_contract.py
checks every public call is listed here and runs the workflow without Qt):

1. Load  [load, peem_service]
    load_beamline(paths, name, pol=...)   UE49 scan tables, .dat or .P3B data -> one dataset (format
                                          from the files; pol = {path: CP/CM/LH/LV} where the data
                                          do not record it). load_ue49 / load_uview: one format.
    find_ue49_scans, set_detector_maps    UE49 helpers (TPX3 dead-pixel / flat-field maps)
    load_path, attach_csv                 TIF stack / folder for the Sum Rule tab, beamline CSV
    unload                                remove a dataset
    get_meta, xmcd_meta                   dataset summary (frames, polarizations, FOV, products ...)

2. Clean the stack (non-destructive, /analysis/stack_edits)  [cleanup]
    set_excluded, include_all, set_reference_frame_op (normalize to / subtract a frame),
    reset_edits, get_edits, edits_summary, get_working_frame,
    disable_pixels, set_topup, injection_frames, frame_sharpness, exclude_defocused

3. Drift correction  [drift_stack]
    drift_correct(name, reg_roi, method, ...)   all frames to one reference -> /processed
    manual_shift, get_drift

4. Stack  [drift_stack]
    stack(name, pair)                     one image per polarization -> /analysis/stack
    stacked_frames                        per polarization and photon energy (energy scans)
    stack_outputs                         [(label, variable)] a front end can show

5. Asymmetry and one-call maps  [drift_stack]
    asymmetry (alias subtract)            (CM-CP)/(CM+CP) or (LV-LH)/(LV+LH) -> /analysis/pcd | pld
    asymmetry_outputs                     [(label, variable)] a front end can show
    compute_pcd, compute_pld, compute_pmld, compute_l2_ld, split_ld_by_reference

6. Viewing and image analysis  [image_analysis]
    view_frames, view_image, map_layers, map_variables, layer_name, parse_layer
    display_hint                          colour scale for a shown image (cmap, symmetric, centre)
    line_profile, stack_line_profiles, histogram, image_fft
    flatten, background_poly, background_roi, fft_filter, use_as_map_data
    get_view_tensor  [peem_service]       one frame of /raw, /processed or a channel

7. Regions and profiles (stored with the data, /analysis/annotations)  [annotations]
    set_roi / get_rois                    "roi", "bg", "drift_box"
    add_profile, list_profiles, clear_profiles
    default_drift_roi  [peem_service]     central default region

8. Maps, ROI statistics, domains  [maps_domains]
    list_products, get_map, get_product_attrs, roi_statistics, auto_fov_roi,
    compute_domain_map, export_map_tiff

9. Sum Rule tab  [drift_stack, peem_service, bg_sumrule]
    to_peem_stack(name, pair, source)     XPEEM-map images -> a new dataset, paired + separated
    prepare_sum_rule                      pair + separate a dataset loaded there
    pair, separate, drift                 the original stack-tab steps (used by prepare_sum_rule)
    bg_preview, bg_apply                  background (linear / two-step, optional ROI)
    sumrule_preview, sumrule_apply        XMCD sum rule

Every step that changes a dataset appends a "PEEM <step>: {parameters}" line to /history; /raw is
never written after loading. Shared helpers: common.py (XPEEM steps), stack_nodes.py (stack nodes).
"""
