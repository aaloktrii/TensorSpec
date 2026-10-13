# UE49 (BESSY II SPEEM) support

Layout: UE49 specifics (`detector.py`, `batch.py`) are here; the beamline-independent analysis
(registration, Igor drift ports, stack clean-up, dichroism, image analysis, ROI stats, domains,
products) is in `core/peem/` next to the original engine / bg / roi / sumrule modules. Together they are numpy ports of the beamline Igor procedures (`speem_eval_20230427_TPX3.ipf`, `ROIAverage.ipf`, `circularcrop*.ipf`), plus a domain map built on the Spatial Domain Analysis (SDA) toolkit. They don't use Qt. They are used by the *XMCD imaging* tab of the PEEM suite (`gui/suites/peem_suite.py`, "PEEM Suite" in the main browser), whose service is the one PEEM service, `gui/services/peem_service.py` (XMCD steps in `gui/services/peem_steps/`).

## Data layout read by `core/io/ue49_loaders.load_scans`

```
<day>/X_0002.txt                     tab table: EnergyTable, ImageNameTable, I0Table, FOVTable, HStab/VStab ...
<day>/X_0002/Cr_pos_0000.tif ...     512x512 TPX3 frames (named by ImageNameTable)
<day>/X_0002/BeamlineSettings.txt    PolTable 0 LH, 1 LV, 2 CP, 3 CM
```

- **Polarization:** a filename token (`pos`/`neg`, `CP`/`CM`/`LH`/`LV`) takes precedence over `PolTable`. On the June 2025 beamtime `PolTable` lagged the helicity switch by one scan: `X_0003` (`Cr_neg`) recorded circular+. Any disagreement goes into `metadata["pol_conflicts"]`. `Mono_S` images are named `Img_xxxx`, so they rely on `PolTable`, and the same lag may apply to them. Check `pol_by_scan` against your logbook.
- **Field of view:** scans with different fields of view (e.g. 15 µm and 10 µm) are refused, because their pixels don't correspond. Pass `allow_mixed_fov=True` only on purpose.

## Detector chain (`detector.tpx3_correct`)

1. **Cross re-gridding:** 512 → 514. The double-width chip-cross pixels are split into 4 lines, then divided by `coef = I_cross / I_neighbours` (3 if the frame is dark).
2. **Orientation:** a diagonal reflection, with `"antitranspose"` as the default. Validated against an Igor-processed co-add (`Cr_neg_0004.tif`, r = 0.987); this orientation also puts the dead-pixel map on the dark pixels.
3. **Dead pixels:** where the map value is 2, the pixel is replaced by the mean of its 4 diagonal neighbours at distance 2.
4. **Flat field:** `img * mean(Norm) / Norm`. There is no dark-current term.
5. **I0:** per-frame `img / I0 * rms(I0)`, applied at load time.

Differences from Igor:
- Arithmetic stays float throughout (Igor truncates to uint16).
- Co-adds are means, not sums, so unequal frame counts don't bias the asymmetry.
- Registration is sub-pixel.

## Outputs under `/analysis/<kind>`

Each output is a Dataset with `data` on (y, x), extra image variables, and scalar attrs.

| kind | definition |
|---|---|
| `pcd` | Circular dichroism. UE49 sign: `(CM−CP)/(CM+CP)`, with `cp_minus_cm` available. CM is registered onto CP using edge images. |
| `pld` | `(LH−LV)/(LH+LV)`. The PMLD step stores it at E_A, with `pld_b` at E_B. |
| `pmld` | Magnetic linear dichroism, the part that reverses between the two L2 multiplet peaks: `[pld(E_A) − pld(E_B)]/2`. |
| `pmnld` | Non-magnetic (natural or crystal-field) linear dichroism, the part common to both peaks: `[pld(E_A) + pld(E_B)]/2`. |
| `domain_map` | Three-state map (+1 up, −1 down, 0 neutral), with variables `m` and `roi`. Its attrs hold f_up, f_down, f_neutral, ⟨m_z⟩, domain counts, widths and period (px and µm), and wall density. |

**Two ways to get pmld / pmnld:**
- *LH/LV route* (`compute_pmld`): pld at the two L2 peaks, split into the part that reverses between them and the part common to both.
- *Single-polarization route* (`compute_l2_ld`, then `split_ld_by_reference`): use this for CrL2_A/CrL2_B series taken at one polarization.
  - First, pld = (I_A − I_B)/(I_A + I_B).
  - The reference is a pld map of the same area taken above T_N. It is registered on intensity images, not on the noisy LD maps, and shifts over 15 % of the image are refused.
  - Then pmnld = reference and pmld = pld − reference.
  - The result reports `noise_ratio` (reference vs map pixel noise, with the smoothing gain computed analytically), `reference_correlation` and `reliable`.
  - On 20250613, the 200 K energy stack (6 frames per energy) is about 5× noisier than the 120-frame 45 K map. It needs `reference_smooth_sigma` of about 4 px, which brings the ratio to 0.35.

**Registration** (`joint_means`):
- All frames of both channels are aligned to one reference, then the second channel is refined with gradient-magnitude images.
- On the 20250613 Cr XMCD repeats this was the most reproducible scheme: repeat correlation 0.78 in the central field, against 0.52 for the Igor maps.
- Per-frame drift agrees with Igor's `displacement_X_0217.txt` to within a median 0.2 px. Co-adds match the Igor `Cr_*_0195` TIFFs at r = 0.998.

**Registration between opposite polarizations** uses gradient-magnitude images. The domain contrast reverses between CP and CM, so plain cross-correlation can lock onto the anti-correlated magnetic pattern.

**`domain_map` classification** follows SDA's `three_state_field` value mode: `m = A/P99|A|`, or the raw A when `normalize="none"`.
- An optional NaN-aware Gaussian smoothing step comes first.
- Despeckling and morphometry use `xmcd_toolkit` when it can be imported, i.e. when it is installed or `TENSORSPEC_SDA_PATH` points at its folder. It needs scikit-image.
- Otherwise a scipy fallback runs. `attrs["backend"]` records which one ran.
- Pass `bg_roi` (a non-magnetic area) to set the zero of the asymmetry. Without it, the +0.01–0.02 helicity offset seen on UE49 maps turns most of the field "up".

## Tests

- `tests/test_ue49_*.py` and `tests/test_peem_xmcd_service.py` use synthetic data with known answers (`core/peem/synthetic.py`).
- `tests/test_peem_realdata.py` runs only when the data folders are set:
  - `UE49_DATA_DIR` = the folder that contains `June 2025/` and `Procedure files/`
  - `TENSORSPEC_BNL_PEEM` = a folder of CP/CM-tagged TIFFs

## Ported since (core/peem/)

- Drift: `autoImageCorrelation`, `driftCorrection`, `marqueeDriftCorrection` (cumulative reference,
  passes, interpolation), `marqueeImageRegistration` (translation + rotation / skew / scale) in `igor_drift.py`.
- Stack clean-up: `RemoveLayerFromStack`, `RemoveDefocussedLayers`, `TopUpCorrection`,
  `NormalizeStackToThisLayer`, `SubtractThisLayerFromStack`, `DisablePixel`, `manualDrift` (`stack_edits.py`).
- Image analysis: `LineProfile`, `getGaussProfile`, `getDiffGaussProfile`, `displayLines` / `stackLines`,
  `calcFrameHistogram`, `imageFFT`, `powerSpectrum`, `FFTFilter`, `Flatten` / `FlattenSequence`,
  `backgroundremoval`, `bkgRemove` (`image_analysis.py`).

## Not ported yet (in the Igor file)

- Vector-map skew correction (`CalcSkew`/`skew`), `Calc2dmag`, angle-series projections. `vector_map` exists but assumes aligned images.
- `RemoveDispersion` / `EnergyShift` (dispersive-plane energy correction).
- `SECO_Fit`, `MergeScans`, `getParticleHyst`, `processDifferential`.
