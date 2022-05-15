import mne

# %%

def spectral_filt(raw, elec_lines, L_cutoff, H_cutoff, ref = ['T9', 'T10', 'Iz']):
    # Remove and Filter signal noises
    raw.notch_filter(elec_lines)
    rawfltrd = raw.filter(L_cutoff, H_cutoff, verbose= False).copy()

    # Referncing to reference electrodes
    rawfltrd = rawfltrd.set_eeg_reference(ref)

    # Check the Power spectral density
    # rawfltrd.plot_psd();
