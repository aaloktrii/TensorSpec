"""MAIN ROUTER for Raman analysis (no Qt).

Like ARPESEngineRouter, band fits are routed by a model key, so fit-model internals stay out
of the UI; like DFTEngineRouter, the other pipeline steps are thin delegating methods. The GUI
(gui/services/raman_service.py, the Raman tabs) and scripts/raman/ talk to this router, not to
core/raman/* fitting modules directly.

Results come back in the form each model produces (dicts / lists of dicts), which the tabs
display as before; core/raman/products.py turns them into xr.Dataset for /analysis.
"""
from __future__ import annotations

from tensorspec.core.raman import (batch, fit_model, multi_lorentz, peak_fit, reference, strain_doping,
                                   tools, voigt_2d)


class RamanEngineRouter:
    """Routes Raman fits and pipeline steps to the core/raman submodules."""

    #: band-fit models and the submodule each one routes to
    MODELS = {
        "voigt": "voigt_2d.fit_band — single Voigt + line (2D-band FWHM / asymmetry maps)",
        "lorentz_n": "multi_lorentz.fit_lorentzians — n Lorentzians + line (Spectra tab)",
        "lorentz_auto": "tools.analyse_band — Si/background, Lorentzians, auto peak count (Peak fitting tab)",
        "g_2d": "peak_fit.analyse_spectrum — G (auto G-/G+ split) and 2D Lorentzians (prediction)",
    }

    # -- band fits (model dispatch) -----------------------------------------------------------
    def fit_band(self, model_choice, wavenumber, intensity, config=None):
        """Fit one spectrum (or band window) with the chosen model.

        voigt         config = the analysis cfg (band window, quality rules, engine)
        lorentz_n     config = {"n": number of Lorentzians}
        lorentz_auto  config = keyword arguments of tools.analyse_band (lo, hi, n, nmax, bg, ...)
        g_2d          config = keyword arguments of peak_fit.analyse_spectrum (ng, n2d, ranges)
        """
        config = config or {}
        if model_choice == "voigt":
            return voigt_2d.fit_band(wavenumber, intensity, config)
        elif model_choice == "lorentz_n":
            return multi_lorentz.fit_lorentzians(wavenumber, intensity, n=int(config.get("n", 1)))
        elif model_choice == "lorentz_auto":
            return tools.analyse_band(wavenumber, intensity, **config)
        elif model_choice == "g_2d":
            return peak_fit.analyse_spectrum(wavenumber, intensity, **config)
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")

    def fit_spectrum(self, wavenumber, intensity, model):
        """Apply the step-1 fit model (Si calibration, background, every band) to one spectrum."""
        return fit_model.fit_spectrum(wavenumber, intensity, model)

    def suggest_peak_count(self, x, y, resolution, nmax=4):
        return multi_lorentz.suggest_peak_count(x, y, resolution, nmax)

    def rebuild_fit(self, x, res):
        """Model curve of a Voigt fit result on x."""
        return voigt_2d.rebuild_fit(x, res)

    def map_fitter(self, mapfile, cfg, progress=None):
        """Voigt fits of every spectrum of a map (2D band, then G when covered): a
        batch.MapFitter whose run() the GUI calls off its main thread."""
        return batch.MapFitter(mapfile, cfg, progress)

    # -- preprocessing -------------------------------------------------------------------------
    def si_shift(self, wavenumber, intensity, ref=tools.SI_REF):
        return tools.si_shift(wavenumber, intensity, ref)

    def noise_level(self, wavenumber, intensity):
        return tools.noise_level(wavenumber, intensity)

    # -- 1L / 2L reference and prediction ------------------------------------------------------
    def load_reference(self, path=reference.DEFAULT_REF):
        return reference.load(path)

    def build_reference(self, refs, out_json, figures=None):
        return batch.build_reference(refs, out_json, figures)

    def predict_row(self, row, ref):
        return reference.predict_row(row, ref)

    def predict_folder(self, folder, ref, out=None, figures=None):
        return batch.predict_folder(folder, ref, out, figures)

    def write_predictions(self, rows, ref, out, title, figures=None):
        return batch.write_predictions(rows, ref, out, title, figures)

    # -- strain / doping -----------------------------------------------------------------------
    def strain_doping_many(self, spectra, ref, workers=None, progress=None):
        return strain_doping.analyse_many(spectra, ref, workers, progress)

    def decompose_strain_doping(self, w_g, w_2d, params=None):
        return strain_doping.decompose(w_g, w_2d, params or strain_doping.PARAMS)
