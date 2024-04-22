import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, "./Lib")

import extract
import bootstrap
import read_hdf
import plot_package


ens_lb = {
    "M1": "chimera_out_48x20x20x20nc4nf2nas3b6.5mf0.71mas1.01_APE0.4N50_smf0.2as0.12_s1",
    "M2": "chimera_out_64x20x20x20nc4nf2nas3b6.5mf0.71mas1.01_APE0.4N50_smf0.2as0.12_s1",
    "M3": "chimera_out_96x20x20x20nc4nf2nas3b6.5mf0.71mas1.01_APE0.4N50_smf0.2as0.12_s1",
    "M4": "chimera_out_64x20x20x20nc4nf2nas3b6.5mf0.70mas1.01_APE0.4N50_smf0.2as0.12_s1",
    "M5": "chimera_out_64x32x32x32nc4nf2nas3b6.5mf0.72mas1.01_APE0.4N50_smf0.24as0.12_s1",
}


CH_bin = {
    "g5": ["g5"],
    "gi": ["g1", "g2", "g3"],
    "g0gi": ["g0g1", "g0g2", "g0g3"],
    "g5gi": ["g5g1", "g5g2", "g5g3"],
    "g0g5gi": ["g0g5g1", "g0g5g2", "g0g5g3"],
    "id": ["id"],
}

DATA = h5py.File("../input_correlators/chimera_data_full.hdf5")


REP = "fund"

ENS = "M1"

ch = "g5"

ti = 13
tf = 23

N_source = 80
N_sink = 40

ens_tag = ens_lb[ENS]
tmp = ens_tag.split("_")[2]
ens = tmp.split("nc")[0] + "b" + tmp.split("b")[1]

Nt = int(tmp.split("x")[0])

print(ens)

ens_group = DATA[ens_tag]

CHs = CH_bin[ch]

tmp_bin = []
for i in range(len(CHs)):
    tmp_bin.append(
        read_hdf.get_meson_corr(
            DATA,
            ens_tag,
            REP,
            N_source,
            N_sink,
            CHs[i],
        )
    )

corr = np.array(tmp_bin).mean(axis=0)

c_boot = bootstrap.Correlator_resample(bootstrap.fold_correlators(corr))

M_tmp = extract.Analysis_Mass_eff_cosh(
    c_boot,
    0,
    Nt,
    ch + " [" + str(N_source) + "," + str(N_sink) + "]",
)


m_tmp, m_tmp_err, chi2 = extract.meson_mass(corr, ti, tf)
plot_package.plot_line(m_tmp, m_tmp_err, ti, tf, plt.gca().lines[-1].get_color())


plt.ylim(0.3, 2)
plt.xlim(0, Nt / 2)

plt.legend()
plt.show()
