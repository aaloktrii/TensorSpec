"""PEEM analysis engines (no Qt).

Original stack workflow          engine (pair / separate / NCC drift), bg (spectral background),
                                 roi (ROI masks), sumrule (XMCD sum rules)
XMCD imaging (UE49 / BNL / ALS)  drift (align_stack: one entry over all drift methods), registration
                                 (sub-pixel FFT, ROI tracking), igor_drift (SPEEM Igor drift ports),
                                 stack_edits (stack clean-up), dichroism (pcd / pld / pmld maps),
                                 image_analysis (profiles, histogram, FFT, flatten, background), roi_stats, domains (SDA), products (/analysis
                                 datasets, ANALYSIS_KINDS), synthetic (test data)
Beamline specifics               ue49/ (TPX3 detector, batch), esm/ (BNL batch)
Readers                          core/io/ (peem_loaders, ue49_loaders, uview_loaders, p3b_loaders,
                                 xpeem_loaders)
"""
