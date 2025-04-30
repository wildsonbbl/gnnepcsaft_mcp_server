"plot utils"

from rdkit.Chem import AllChem as Chem

from .utils import smilestoinchi


def v3000_mol_block(smiles: str) -> str:
    """Returns a V3000 Mol block for a molecule.

    Args:
        smiles (str): SMILES of the molecule.

    """
    inchi = smilestoinchi(smiles)

    mol = Chem.MolFromInchi(inchi)
    mol = Chem.AddHs(mol)  # type: ignore
    params = Chem.ETKDGv3()  # type: ignore
    params.randomSeed = 0xF00D
    result = Chem.EmbedMolecule(mol, params)  # type: ignore
    if result == 0:
        Chem.MMFFOptimizeMolecule(  # type: ignore
            mol, maxIters=1000, nonBondedThresh=100, ignoreInterfragInteractions=False
        )
    # mol = Chem.RemoveHs(mol, implicitOnly=False)
    imgmol = Chem.MolToV3KMolBlock(mol)  # type: ignore
    return imgmol
