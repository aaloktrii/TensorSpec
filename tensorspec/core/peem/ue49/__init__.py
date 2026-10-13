"""BESSY II UE49 SPEEM specifics (ports of the beamline Igor macros, no Qt).

Modules
-------
detector      TPX3 512->514 cross correction, dead-pixel / flat-field, dark, I0, crops
batch         logbook-driven beamtime processing (scripts/ue49/process_beamtime.py)

Loading: core/io/ue49_loaders.py (any format: core/io/xpeem_loaders.py).
Beamline-independent analysis (registration, dichroism, drift, image analysis, domains, ...):
core/peem/ (registration, dichroism, igor_drift, image_analysis, stack_edits, ...).
"""
