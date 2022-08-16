import rdkit.Chem as Chem
from rdkit import DataStructs
import pandas as pd
from rdkit.Chem import AllChem
import rdkit.Chem.Descriptors as desc
import rdkit.Chem.rdMolDescriptors as Rdesc
from rdkit.Chem import rdmolfiles
import rdkit
import numpy as np


mw = desc.ExactMolWt
fpdensity1 = desc.FpDensityMorgan1
fpdensity2 = desc.FpDensityMorgan2
fpdensity3 = desc.FpDensityMorgan3
heavy_atom_wt = desc.HeavyAtomMolWt
max_abs_pt_chg = desc.MaxAbsPartialCharge
max_ptl_chg = desc.MaxPartialCharge
min_abs_chg = desc.MinAbsPartialCharge
min_ptl_chg = desc.MinPartialCharge
avg_mol_wt = desc.MolWt
num_radical = desc.NumRadicalElectrons
num_valence = desc.NumValenceElectrons
# bcut2d = Rdesc.BCUT2D #vector
calcauto2d = Rdesc.CalcAUTOCORR2D #vector
chi0n = Rdesc.CalcChi0n
chi0v = Rdesc.CalcChi0v
chi1n = Rdesc.CalcChi1n
chi1v = Rdesc.CalcChi1v
chi2n = Rdesc.CalcChi2n
chi2v = Rdesc.CalcChi2v
chi3n = Rdesc.CalcChi3n
chi3v = Rdesc.CalcChi3v
chi4n = Rdesc.CalcChi4n
chi4v = Rdesc.CalcChi4v
crippen = Rdesc.CalcCrippenDescriptors #tuple
frac_C_sp3 = Rdesc.CalcFractionCSP3
hall_kier = Rdesc.CalcHallKierAlpha
kapp1 = Rdesc.CalcKappa1
kappa2 = Rdesc.CalcKappa2
kappa3 = Rdesc.CalcKappa3
labute_asa = Rdesc.CalcLabuteASA
alphaptic_carbo = Rdesc.CalcNumAliphaticCarbocycles
alaphatic_hetero = Rdesc.CalcNumAliphaticHeterocycles
rings_ala = Rdesc.CalcNumAliphaticRings
amide_bonds = Rdesc.CalcNumAmideBonds
aromatic_carbo = Rdesc.CalcNumAromaticCarbocycles
aromatic_hetero = Rdesc.CalcNumAromaticHeterocycles
rings_aromatic = Rdesc.CalcNumAromaticRings
stereo_centers = Rdesc.CalcNumAtomStereoCenters
bridge_head = Rdesc.CalcNumBridgeheadAtoms
hba = Rdesc.CalcNumHBA
hbd = Rdesc.CalcNumHBD
hetero_atoms = Rdesc.CalcNumHeteroatoms
heterocycles = Rdesc.CalcNumHeterocycles
lipinski_hba  = Rdesc.CalcNumLipinskiHBA
lipinski_hbd = Rdesc.CalcNumLipinskiHBD
num_rings = Rdesc.CalcNumRings
rotatable_bonds = Rdesc.CalcNumRotatableBonds
saturated_carbo = Rdesc.CalcNumSaturatedCarbocycles
sat_hetero = Rdesc.CalcNumSaturatedHeterocycles
sat_rings = Rdesc.CalcNumSaturatedRings
unspecified_stereo = Rdesc.CalcNumUnspecifiedAtomStereoCenters
tpsa = Rdesc.CalcTPSA
mqns = Rdesc.MQNs_ #list
peoe = Rdesc.PEOE_VSA_
color = rdmolfiles.CanonicalRankAtoms
bond_type = Chem.rdchem.Bond.GetBondType
triple = rdkit.Chem.rdchem.BondType.TRIPLE
double = rdkit.Chem.rdchem.BondType.DOUBLE
single = rdkit.Chem.rdchem.BondType.SINGLE
aromatic = rdkit.Chem.rdchem.BondType.AROMATIC
get_1024_morgan_bit = AllChem.GetMorganFingerprintAsBitVect
to = Chem.MolToSmiles

max_d = pd.read_csv("ocelotml_2d/normalized_feats.csv", index_col = 0)
#max_d = pd.read_csv("normalized_feats.csv", index_col = 0)
max_d_arr = np.array(max_d).transpose()

def all_properties(mol):
    smile = to(mol)
    count = smile.count
    prop_dict = dict()
    bonds = [bond_type(bond) for bond in mol.GetBonds()]
    prop_dict["num_single"] = bonds.count(single)
    prop_dict["num_double"] = bonds.count(double)
    prop_dict["num_triple"] = bonds.count(triple)
    prop_dict["aromatic_bonds"] = bonds.count(aromatic)
    prop_dict["num_fluorine"] = count("F")
    prop_dict["num_chlorine"] = count("Cl")
    prop_dict["num_bromine"] = count("Br")
    prop_dict["num_nitro"] = count("N") + count("n")
    prop_dict["num_oxy"] = count("O") + count("o")
    prop_dict["num_sulfur"] = count("S") + count("s") -count("Se")-count("se")
    prop_dict["unique_colors"] = len(set(color(mol)))
    m_peoe = peoe(mol)
    for i, mp in enumerate(m_peoe):
        prop_dict[F"peoe_{i}"] = mp
    m_mqns = mqns(mol)
    for j, mq in enumerate(m_mqns):
        prop_dict[F"mqns_{i}"] = mq
    prop_dict["tpsa"] = tpsa(mol)
    prop_dict["unspecified_stero"] = unspecified_stereo(mol)
    prop_dict["sat_rings"] = sat_rings(mol)
    prop_dict["sat_het"] = sat_hetero(mol)
    prop_dict["sat_carbo"] = saturated_carbo(mol)
    prop_dict["rotable_bonds"] = rotatable_bonds(mol)
    prop_dict["num_rings"] = num_rings(mol)
    prop_dict["lipinski_hba"] = lipinski_hba(mol)
    prop_dict["lipinski_hbd"] = lipinski_hbd(mol)
    prop_dict["hetero"] = heterocycles(mol)
    prop_dict["hetero_atoms"] = hetero_atoms(mol)
    prop_dict["h_accept"] = hba(mol)
    prop_dict["donate_h"] = hbd(mol)
    prop_dict["bridge_head"] = bridge_head(mol)
    prop_dict["stereo"] = stereo_centers(mol)
    prop_dict["aromatic_rings"] = rings_aromatic(mol)
    prop_dict["hetero_aromatic"] = aromatic_hetero(mol)
    prop_dict["carbo_aromatic"] = aromatic_carbo(mol)
    prop_dict["num_amide"] = amide_bonds(mol)
    prop_dict["ala_rings"] = rings_ala(mol)
    prop_dict["alaphatic_hetero"] = alaphatic_hetero(mol)
    prop_dict["alaphatic_carbo"] = alphaptic_carbo(mol)
    prop_dict["labuta_asa"] = labute_asa(mol)
    prop_dict["kappa1"] = kapp1(mol)
    prop_dict["kappa2"] = kappa2(mol)
    prop_dict["kappa3"] = kappa3(mol)
    crip = crippen(mol)
    for k, cri in enumerate(crip):
        prop_dict[F"criippen_{k}"] = cri
    prop_dict["c_sp3"] = frac_C_sp3(mol)
    prop_dict["hall_kier"] = hall_kier(mol)
    prop_dict["chi0n"] = chi0n(mol)
    prop_dict["chi0v"] = chi0v(mol)
    prop_dict["chi1n"] = chi1n(mol)
    prop_dict["chi1v"] = chi1v(mol)
    prop_dict["chi2n"] = chi2n(mol)
    prop_dict["chi2v"] = chi2v(mol)
    prop_dict["chi3n"] = chi3n(mol)
    prop_dict["chi3v"] = chi3v(mol)
    prop_dict["chi4n"] = chi4n(mol)
    prop_dict["chi4v"] = chi4v(mol)
    auto_2d = calcauto2d(mol)
    for l, _2d in enumerate(auto_2d):
        prop_dict[F"auto_2d_{l}"] = _2d

    prop_dict["num_valence"] = num_valence(mol)
    prop_dict["num_radical"] = num_radical(mol)
    prop_dict["avg_mol_weight"] = avg_mol_wt(mol)
    prop_dict["heavy_atom"] = heavy_atom_wt(mol)
    prop_dict["fp1_densirt"] = fpdensity1(mol)
    prop_dict["fp2_density"] = fpdensity2(mol)
    prop_dict["fp3_density"] = fpdensity3(mol)
    prop_dict["weight"] = mw(mol)
    return prop_dict


def molecule_descriptors(mol, fp=False):
    one = []
    if fp:
        fingerprint = get_1024_morgan_bit(mol, 2, nBits = fp)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        one = array.tolist()
    info = all_properties(mol)
    info = list(info.values())
    two = np.array(info)
    two = two/max_d_arr
    return one + two.tolist()[0]
