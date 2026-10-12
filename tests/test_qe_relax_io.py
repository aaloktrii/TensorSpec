from pymatgen.core import Lattice, Structure
from tensorspec.core.dft.qe_relax_io import parse_qe_relaxed_structure, write_relaxed_cif

FIXTURE = """
     site n.     atom                  positions (alat units)
         1           C   tau(   1 ) = (  0.3333333  0.6666667  0.4000000  )
         2           C   tau(   2 ) = (  0.6666667  0.3333333  0.4000000  )
         3           B   tau(   3 ) = (  0.3333333  0.6666667  0.6000000  )
         4           N   tau(   4 ) = (  0.6666667  0.3333333  0.6000000  )
"""

# Prefer parser that reads final ATOMIC_POSITIONS crystal block if present:
FIXTURE_CRYSTAL = """
ATOMIC_POSITIONS (crystal)
C  0.333333  0.666667  0.410000
C  0.666667  0.333333  0.410000
B  0.333333  0.666667  0.590000
N  0.666667  0.333333  0.590000
End final coordinates
"""


def test_parse_final_crystal_positions():
    lat = Lattice.hexagonal(2.5, 23.4)
    template = Structure(lat, ["C", "C", "B", "N"],
                         [[1/3, 2/3, 0.42], [2/3, 1/3, 0.42],
                          [1/3, 2/3, 0.58], [2/3, 1/3, 0.58]])
    s = parse_qe_relaxed_structure(FIXTURE_CRYSTAL, template)
    assert abs(s[0].frac_coords[2] - 0.41) < 1e-5
    assert s.lattice.a == template.lattice.a  # ionic relax: cell unchanged


def test_write_relaxed_cif(tmp_path):
    lat = Lattice.hexagonal(2.5, 23.4)
    s = Structure(lat, ["C", "C"], [[1/3, 2/3, 0.4], [2/3, 1/3, 0.4]])
    path = write_relaxed_cif(s, str(tmp_path / "relaxed_structure.cif"))
    s2 = Structure.from_file(path)
    assert len(s2) == 2
