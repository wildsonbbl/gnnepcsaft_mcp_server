"tests"

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# pylint: disable=too-few-public-methods,unused-argument,wrong-import-position
# pylint: disable=too-many-arguments
# pylint: disable=unnecessary-lambda

import sys
import unittest
from unittest.mock import MagicMock, patch

sys.modules["polars"] = MagicMock()

# Mock thermodynamic backend libraries
sys.modules["gnnepcsaft"] = MagicMock()
sys.modules["gnnepcsaft.data.ogb_utils"] = MagicMock()
sys.modules["gnnepcsaft.data.rdkit_util"] = MagicMock()
sys.modules["gnnepcsaft.pcsaft"] = MagicMock()
sys.modules["gnnepcsaft.pcsaft.pcsaft_feos"] = MagicMock()

# -- IMPORT MODULES TO TEST --
from gnnepcsaft_mcp_server import utils_mix, utils_pure


class TestUtilsPure(unittest.TestCase):
    "test utils_pure.py"

    @patch("gnnepcsaft_mcp_server.utils_pure.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_pure.pure_den_feos")
    def test_pure_den(self, mock_calc, mock_predict):
        """Test Pure Density Logic"""
        # Setup mocks
        mock_predict.return_value = "dummy_params"
        mock_calc.return_value = 1000.0  # Mocked density result

        # Execute
        temps, dens = utils_pure.pure_den("water", 300, 310, 101325, 10)

        # Assert
        self.assertEqual(len(temps), 10)  # np.linspace with num=10
        self.assertEqual(len(dens), 10)
        self.assertEqual(dens[0], 1000.0)
        mock_predict.assert_called_with("water")

    @patch("gnnepcsaft_mcp_server.utils_pure.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_pure.pure_vp_feos")
    def test_pure_vp(self, mock_calc, mock_predict):
        """Test Pure Vapor Pressure Logic"""
        mock_predict.return_value = "dummy_params"
        mock_calc.return_value = 12345.0

        temps, vps = utils_pure.pure_vp("ethanol", 300, 310, 10)

        self.assertEqual(len(temps), 10)
        self.assertEqual(vps[0], 12345.0)


class TestUtilsMix(unittest.TestCase):
    "test utils_mix.py"

    @patch("gnnepcsaft_mcp_server.utils_mix.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_mix.mix_den_feos")
    def test_mix_den(self, mock_calc, mock_predict):
        """Test Mixture Density Logic"""
        mock_predict.side_effect = ["p1", "p2"]
        mock_calc.return_value = 800.0

        smiles = ["C1", "C2"]
        fracs = [0.5, 0.5]
        kij = [[0.0, 0.0], [0.0, 0.0]]
        params = utils_mix.MixDenParams(
            smiles_list=smiles,
            mole_fractions=fracs,
            kij_matrix=kij,
            min_temp=300,
            max_temp=310,
            pressure=100000,
            npoints=10,
        )
        temps, dens = utils_mix.mix_den(params)

        self.assertEqual(len(temps), 10)
        self.assertEqual(dens[0], 800.0)

        # Verify call arguments structure
        call_kwargs = mock_calc.call_args[1]
        self.assertIn("parameters", call_kwargs)
        self.assertIn("state", call_kwargs)
        self.assertIn("kij_matrix", call_kwargs)

    @patch("gnnepcsaft_mcp_server.utils_mix.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_mix.mix_vle_diagram_feos")
    def test_mix_vle(self, mock_calc, mock_predict):
        """Test Mixture VLE Logic"""
        mock_predict.return_value = "p"
        expected_output = {"x0": [0.1], "y0": [0.9], "temperature": [300]}
        mock_calc.return_value = expected_output

        res = utils_mix.mix_vle(["A", "B"], [[0, 0], [0, 0]], 101325, 10)

        self.assertEqual(res, expected_output)


if __name__ == "__main__":
    unittest.main()
