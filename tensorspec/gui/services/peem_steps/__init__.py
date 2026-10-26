"""PEEM back end: the calls any front end uses (Qt panels, scripts, a web page).

``PeemService`` (gui/services/peem_service.py) is the one PEEM service: the original load / pair /
drift / BG / sum-rule methods (in peem_service.py) plus the steps in this package (mixins). Front
ends: the PEEM panel (load, pair, drift, then calculate and analyse through
gui/components/peem_map_tools.py) and the Sum Rule window (XAS suite, From PEEM tab:
gui/components/peem_sumrule_panel.py). The service holds no front-end state:
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
    load_path, attach_csv  [peem_service] TIF stack / folder for the PEEM tab, beamline CSV
    inspect_folder, load_images           "Load Folder": what a folder holds (UE49 / UView / P3B / TIF /
                                          PNG / JPEG / a folder of acquisitions); PNG / JPEG frames
    load_maps_nc(path, name)              a NetCDF map file (batch maps.nc, analysis outputs, any 2-D
                                          variables) -> dataset: /analysis/nc (every map, each with its
                                          type signed / intensity / domain), + stack and pcd / pld when
                                          the file has channel means; nc_outputs, map_type for display
    list_datasets                         loaded PEEM datasets (the panel's Dataset list)
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
    view_frames, view_image, map_layers, shown_layers, map_variables, layer_name, layer_for, parse_layer
                                          (shown_layers: map_layers less sums, copies, working arrays)
                                          (the stack's CP + CM image is the layer 'stack · cp+cm')
    display_hint                          colour scale for a shown image (cmap, symmetric, centre)
    line_profile, stack_line_profiles, histogram, image_fft
    flatten, background_poly, background_roi, fft_filter, use_as_map_data
    get_view_tensor  [peem_service]       one frame of /raw, /processed or a channel

7. Regions and profiles (stored with the data, /analysis/annotations)  [annotations]
    set_roi / get_rois                    "roi", "bg", "drift_box"
    add_profile, list_profiles, clear_profiles, remove_profile, set_profile_visible (show / hide)
    default_drift_roi  [peem_service]     central default region

8. Maps, ROI statistics, domains  [maps_domains]
    list_products, get_map, get_product_attrs, roi_statistics, auto_fov_roi,
    compute_domain_map, export_map_tiff (float TIFF of the values + colour PNG to look at)
    export_options, export_map            export with options: crop, picture size (px) / dpi,
                                          PNG / PDF / SVG, colours and limits, colour bar, scale bar

9. Hand-over of frames to a new paired dataset  [drift_stack]
    to_peem_stack(name, pair, source)     drift-corrected frames -> a new dataset, paired + separated

10. PEEM panel -> maps and the Sum Rule window  [channels]
    channel_stack(name, e_lo, e_hi)       the panel's corrected CP / CM pairs (after Stack Pairs and
                                          Drift) -> one image each -> /analysis/stack; then
                                          asymmetry, image analysis, ROIs, domains, export (5.-8.)
    domain_width(name, kind, var, roi)    mean domain width = area / domain-wall length
    to_sum_rule(name)                     corrected pairs -> new dataset <name>_sumrule with an energy
                                          per pair, paired + separated, for background / sum rule
    sum_rule_datasets, channel_image      the Sum Rule window's dataset list and region image
    roi_spectra(name, roi)                CP / CM spectra of a region
    map_to_tools(name, kind)              a section-4 map (Make PCD / PLD) -> /analysis/stack + rebuilt
                                          /analysis/<kind>, so image analysis, ROIs and Compare use it

11. Compare with a reference dataset  [compare]
    compare_candidates(name)              other loaded datasets with an asymmetry map
    compare_maps(name, reference, kind, search, rotation, smooth_reference)
                                          the reference's map registered onto this one on the intensity
                                          images (core/peem/map_align: shift, rotation, scale; search
                                          'auto' / 'none' / 'full' for re-imaged fields), subtracted ->
                                          /analysis/compare (data = map - reference, map, reference),
                                          e.g. XMLD = XLD(T) - XLD(T > T_c)
    compare_outputs, export_compare_figure   what to show; map | reference | difference, one grey scale

Original PEEM steps (peem_service.py): pair, separate, drift, get_view_tensor, default_drift_roi;
background and sum rule (bg_preview, bg_apply, sumrule_preview, sumrule_apply, with use_roi /
roi_dict for a region), used by the Sum Rule window.
Dichroism map on the panel (peem_service.py, section 4): make_pcd_map, make_pld_map ->
/analysis/pcd | pld; list_analysis_maps, get_display_tensor (any node for the Frame viewer),
export_analysis_map (TIFF + colour PNG).

Every step that changes a dataset appends a "PEEM <step>: {parameters}" line to /history; /raw is
never written after loading. Shared helpers of the steps: common.py.
"""
