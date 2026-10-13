"""NSLS-II ESM (21-ID) PEEM support: Elmitec UView .dat images and the
beamtime folder layout used at BNL (no Qt).

reader  core/io/uview_loaders.py (UKSOFT2001 .dat; image + sample temperature,
        vacuum gauges, start voltage, FOV, timestamp; CL/CR/LH/LV folders)
batch   folder-name catalog (XPEEM_<FOV>um_..._<E>eV_Cr2p_<CL|CR|LH|LV>_<az>deg[_n]),
        XMCD / XLD units, repeat clusters, RT-vs-low-T LD split, XAS, LEEM previews
"""
