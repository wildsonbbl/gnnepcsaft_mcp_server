"tests"

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access
# pylint: disable=too-few-public-methods,unused-argument,wrong-import-position
# pylint: disable=too-many-arguments,too-many-public-methods
# pylint: disable=unnecessary-lambda

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules["polars"] = MagicMock()

# Mock thermodynamic backend libraries
sys.modules["gnnepcsaft"] = MagicMock()
sys.modules["gnnepcsaft.data.ogb_utils"] = MagicMock()
sys.modules["gnnepcsaft.data.rdkit_util"] = MagicMock()
sys.modules["gnnepcsaft.pcsaft"] = MagicMock()
sys.modules["gnnepcsaft.pcsaft.pcsaft_feos"] = MagicMock()

# -- IMPORT MODULES TO TEST --
from gnnepcsaft_mcp_server import utils_mix, utils_pure
from gnnepcsaft_mcp_server.utils import (
    batch_convert_pure_density_to_kg_per_m3,
    batch_critical_points,
    batch_inchi_to_smiles,
    batch_molecular_weights,
    batch_pa_to_bar,
    batch_predict_pcsaft_parameters,
    batch_pure_density,
    batch_pure_h_lv,
    batch_pure_vapor_pressure,
    batch_smiles_to_inchi,
    mixture_density,
    mixture_phase,
    mixture_vapor_pressure,
    predict_pcsaft_parameters,
    pubchem_description,
    pure_phase,
)

METHANE_SMILES = "C"
ETHANOL_SMILES = "CCO"
WATER_SMILES = "O"
TEST_SMILES_LIST = [METHANE_SMILES, ETHANOL_SMILES, WATER_SMILES]


class TestUtilsCore(unittest.TestCase):
    "Test direct utility functions from utils.py"

    @patch("gnnepcsaft_mcp_server.utils.mw", return_value=16.04)
    @patch("gnnepcsaft_mcp_server.utils.assoc_number", return_value=(0, 0))
    @patch("gnnepcsaft_mcp_server.utils.smiles2graph")
    @patch(
        "gnnepcsaft_mcp_server.utils.smilestoinchi", return_value="InChI=1S/CH4/h1H4"
    )
    @patch("gnnepcsaft_mcp_server.utils.msigmae_onnx")
    @patch("gnnepcsaft_mcp_server.utils.assoc_onnx")
    def test_predict_pcsaft_parameters(
        self,
        mock_assoc_onnx,
        mock_msigmae_onnx,
        mock_smilestoinchi,
        mock_smiles2graph,
        mock_assoc_number,
        mock_mw,
    ):
        mock_smiles2graph.return_value = {
            "node_feat": np.zeros((1, 1)),
            "edge_index": np.array([[0], [0]]),
            "edge_feat": np.zeros((1, 1)),
        }
        mock_assoc_onnx.run.return_value = [np.array([0.0, 0.0])]
        mock_msigmae_onnx.run.return_value = [np.array([[1.0, 3.7, 150.0]])]

        params = predict_pcsaft_parameters(METHANE_SMILES)
        self.assertIsInstance(params, list)
        self.assertEqual(len(params), 9)
        self.assertTrue(all(isinstance(p, float) for p in params))

    @patch("gnnepcsaft_mcp_server.utils.predict_pcsaft_parameters")
    def test_batch_predict_pcsaft_parameters(self, mock_predict):
        mock_predict.side_effect = [
            [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.04],
            [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 46.07],
            [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.02],
        ]

        params_list = batch_predict_pcsaft_parameters(TEST_SMILES_LIST)
        self.assertIsInstance(params_list, list)
        self.assertEqual(len(params_list), len(TEST_SMILES_LIST))
        self.assertTrue(all(len(params) == 9 for params in params_list))

    def test_pure_phase_liquid(self):
        self.assertEqual(
            pure_phase(vapor_pressure=90000, system_pressure=100000), "liquid"
        )

    def test_pure_phase_vapor(self):
        self.assertEqual(
            pure_phase(vapor_pressure=110000, system_pressure=100000), "vapor"
        )

    def test_pure_phase_validation(self):
        with pytest.raises(AssertionError):
            pure_phase(-1, 100000)
        with pytest.raises(AssertionError):
            pure_phase(100000, -1)
        with pytest.raises(AssertionError):
            pure_phase("invalid", 100000)  # type: ignore[arg-type]

    def test_mixture_phase_liquid(self):
        self.assertEqual(
            mixture_phase(bubble_point=90000, dew_point=120000, system_pressure=100000),
            "liquid",
        )

    def test_mixture_phase_vapor(self):
        self.assertEqual(
            mixture_phase(
                bubble_point=110000, dew_point=120000, system_pressure=100000
            ),
            "vapor",
        )

    def test_mixture_phase_two_phase(self):
        self.assertEqual(
            mixture_phase(bubble_point=110000, dew_point=90000, system_pressure=100000),
            "two-phase",
        )

    def test_mixture_phase_validation(self):
        with pytest.raises(AssertionError):
            mixture_phase(-1, 100000, 100000)
        with pytest.raises(AssertionError):
            mixture_phase(100000, -1, 100000)
        with pytest.raises(AssertionError):
            mixture_phase(100000, 100000, -1)

    @patch(
        "gnnepcsaft_mcp_server.utils.smilestoinchi", return_value="InChI=1S/CH4/h1H4"
    )
    @patch("gnnepcsaft_mcp_server.utils.urlopen")
    def test_pubchem_description_success(self, mock_urlopen, mock_smilestoinchi):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"Test": "Data"}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = pubchem_description(METHANE_SMILES)
        self.assertEqual(result, {"Test": "Data"})

    @patch("gnnepcsaft_mcp_server.utils.urlopen")
    def test_pubchem_description_error(self, mock_urlopen):
        mock_urlopen.side_effect = ValueError()

        result = pubchem_description(METHANE_SMILES)
        self.assertEqual(result, "no data available on this molecule in PubChem.")

    @patch("gnnepcsaft_mcp_server.utils.mw", side_effect=[16.04, 46.07, 18.02])
    @patch(
        "gnnepcsaft_mcp_server.utils.smilestoinchi",
        side_effect=[
            "InChI=1S/CH4/h1H4",
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            "InChI=1S/H2O/h1H2",
        ],
    )
    def test_batch_molecular_weights(self, mock_smilestoinchi, mock_mw):
        weights = batch_molecular_weights(TEST_SMILES_LIST)
        self.assertIsInstance(weights, list)
        self.assertEqual(len(weights), len(TEST_SMILES_LIST))
        self.assertTrue(all(isinstance(w, float) for w in weights))
        self.assertTrue(15.5 < weights[0] < 16.5)

    @patch("gnnepcsaft_mcp_server.utils.inchitosmiles", side_effect=["C", "CCO", "O"])
    @patch(
        "gnnepcsaft_mcp_server.utils.smilestoinchi",
        side_effect=[
            "InChI=1S/CH4/h1H4",
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            "InChI=1S/H2O/h1H2",
        ],
    )
    def test_batch_inchi_to_smiles(self, mock_smilestoinchi, mock_inchitosmiles):
        inchi_list = batch_smiles_to_inchi(TEST_SMILES_LIST)
        smiles_list = batch_inchi_to_smiles(inchi_list)
        self.assertEqual(len(smiles_list), len(TEST_SMILES_LIST))
        self.assertTrue(all(isinstance(s, str) for s in smiles_list))

    @patch(
        "gnnepcsaft_mcp_server.utils.smilestoinchi",
        side_effect=[
            "InChI=1S/CH4/h1H4",
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            "InChI=1S/H2O/h1H2",
        ],
    )
    def test_batch_smiles_to_inchi(self, mock_smilestoinchi):
        inchi_list = batch_smiles_to_inchi(TEST_SMILES_LIST)
        self.assertIsInstance(inchi_list, list)
        self.assertEqual(len(inchi_list), len(TEST_SMILES_LIST))
        self.assertTrue(all(isinstance(i, str) for i in inchi_list))
        self.assertTrue(all(i.startswith("InChI=") for i in inchi_list))

    @patch("gnnepcsaft_mcp_server.utils.mix_den_feos", return_value=850.0)
    def test_mixture_density(self, mock_mix_den):
        parameters = [
            [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.04],
            [2.0, 3.5, 200.0, 0.01, 2000.0, 0.0, 1.0, 1.0, 46.07],
        ]
        state = [298.15, 101325, 0.5, 0.5]
        kij_matrix = [[0.0, 0.0], [0.0, 0.0]]

        density = mixture_density(parameters, state, kij_matrix)
        self.assertIsInstance(density, float)
        self.assertGreater(density, 0)

    @patch("gnnepcsaft_mcp_server.utils.mix_vp_feos", return_value=(120000.0, 90000.0))
    def test_mixture_vapor_pressure(self, mock_mix_vp):
        parameters = [
            [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.04],
            [2.0, 3.5, 200.0, 0.01, 2000.0, 0.0, 1.0, 1.0, 46.07],
        ]
        state = [298.15, 0.0, 0.5, 0.5]
        kij_matrix = [[0.0, 0.0], [0.0, 0.0]]

        bubble_point, dew_point = mixture_vapor_pressure(parameters, state, kij_matrix)
        self.assertIsInstance(bubble_point, float)
        self.assertIsInstance(dew_point, float)
        self.assertGreater(bubble_point, 0)
        self.assertGreater(dew_point, 0)

    @patch("gnnepcsaft_mcp_server.utils.pure_den_feos")
    @patch("gnnepcsaft_mcp_server.utils.predict_pcsaft_parameters")
    def test_batch_pure_density(self, mock_predict, mock_den_feos):
        mock_predict.return_value = [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_den_feos.return_value = 500.0

        state = [298.15, 101325]
        densities = batch_pure_density(TEST_SMILES_LIST, state)

        self.assertIsInstance(densities, list)
        self.assertEqual(len(densities), len(TEST_SMILES_LIST))
        self.assertTrue(all(d == 500.0 for d in densities))

    @patch("gnnepcsaft_mcp_server.utils.pure_vp_feos")
    @patch("gnnepcsaft_mcp_server.utils.predict_pcsaft_parameters")
    def test_batch_pure_vapor_pressure(self, mock_predict, mock_vp_feos):
        mock_predict.return_value = [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_vp_feos.return_value = 50000.0

        temperature = 298.15
        pressures = batch_pure_vapor_pressure(TEST_SMILES_LIST, temperature)

        self.assertIsInstance(pressures, list)
        self.assertEqual(len(pressures), len(TEST_SMILES_LIST))
        self.assertTrue(all(p == 50000.0 for p in pressures))

    @patch("gnnepcsaft_mcp_server.utils.pure_h_lv_feos")
    @patch("gnnepcsaft_mcp_server.utils.predict_pcsaft_parameters")
    def test_batch_pure_h_lv(self, mock_predict, mock_h_lv_feos):
        mock_predict.return_value = [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_h_lv_feos.return_value = 20.0

        temperature = 298.15
        enthalpies = batch_pure_h_lv(TEST_SMILES_LIST, temperature)

        self.assertIsInstance(enthalpies, list)
        self.assertEqual(len(enthalpies), len(TEST_SMILES_LIST))
        self.assertTrue(all(h == 20.0 for h in enthalpies))

    @patch("gnnepcsaft_mcp_server.utils.critical_points_feos")
    @patch("gnnepcsaft_mcp_server.utils.predict_pcsaft_parameters")
    def test_batch_critical_points(self, mock_predict, mock_critical_points):
        mock_predict.return_value = [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_critical_points.return_value = [190.0, 4600000.0, 10200.0]

        critical_points = batch_critical_points(TEST_SMILES_LIST)

        self.assertIsInstance(critical_points, list)
        self.assertEqual(len(critical_points), len(TEST_SMILES_LIST))
        self.assertTrue(all(len(cp) == 3 for cp in critical_points))

    def test_batch_pa_to_bar(self):
        pa_values = [100000.0, 200000.0, 300000.0]
        bar_values = batch_pa_to_bar(pa_values)
        self.assertEqual(bar_values, [1.0, 2.0, 3.0])

    def test_batch_convert_pure_density_to_kg_per_m3(self):
        density_values = [1000.0, 2000.0, 3000.0]
        mw_values = [16.04, 46.07, 18.02]
        kg_per_m3_values = batch_convert_pure_density_to_kg_per_m3(
            density_values, mw_values
        )

        self.assertIsInstance(kg_per_m3_values, list)
        self.assertEqual(len(kg_per_m3_values), len(density_values))
        self.assertEqual(
            kg_per_m3_values,
            [den * mw / 1000 for den, mw in zip(density_values, mw_values)],
        )


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

    @patch("gnnepcsaft_mcp_server.utils_pure.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_pure.pure_h_lv_feos")
    def test_pure_h_lv(self, mock_calc, mock_predict):
        """Test Pure Component Enthalpy of Vaporization"""
        mock_predict.return_value = "dummy_params"
        mock_calc.return_value = 40660.0  # kJ/mol for water

        temps, h_lvs = utils_pure.pure_h_lv("water", 300, 310, 10)

        self.assertEqual(len(temps), 10)
        self.assertEqual(len(h_lvs), 10)
        self.assertEqual(h_lvs[0], 40660.0)
        mock_predict.assert_called_with("water")

    @patch("gnnepcsaft_mcp_server.utils_pure.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_pure.pure_s_lv_feos")
    def test_pure_s_lv(self, mock_calc, mock_predict):
        """Test Pure Component Entropy of Vaporization"""
        mock_predict.return_value = "dummy_params"
        mock_calc.return_value = 118.9  # J/(mol·K) for water

        temps, s_lvs = utils_pure.pure_s_lv("water", 300, 310, 10)

        self.assertEqual(len(temps), 10)
        self.assertEqual(len(s_lvs), 10)
        self.assertEqual(s_lvs[0], 118.9)

    @patch("gnnepcsaft_mcp_server.utils_pure.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_pure.pure_surface_tension_feos")
    def test_pure_surface_tension(self, mock_calc, mock_predict):
        """Test Pure Component Surface Tension"""
        mock_predict.return_value = "dummy_params"
        # pure_surface_tension_feos returns (surface_tensions, temperatures)
        mock_calc.return_value = (np.array([72.0]), np.array([298.15]))

        temps, sts = utils_pure.pure_surface_tension("water", 298.15)

        self.assertEqual(len(temps), 1)
        self.assertEqual(len(sts), 1)
        self.assertEqual(sts[0], 72.0)

    @patch("gnnepcsaft_mcp_server.utils_pure.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_pure.phase_diagram_feos")
    def test_pure_phase_diagram(self, mock_calc, mock_predict):
        """Test Pure Component Phase Diagram"""
        mock_predict.return_value = "dummy_params"
        mock_calc.return_value = {
            "temperature": [300.0, 310.0],
            "pressure": [100.0, 200.0],
            "density liquid": [800.0, 790.0],
            "density vapor": [1.0, 1.5],
        }

        temps, pressures, rho_l, rho_v = utils_pure.pure_phase_diagram("water", 300.0)

        self.assertEqual(len(temps), 2)
        self.assertEqual(len(pressures), 2)
        self.assertEqual(len(rho_l), 2)
        self.assertEqual(len(rho_v), 2)
        self.assertEqual(temps[0], 300.0)
        self.assertEqual(pressures[0], 100.0)


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

    @patch("gnnepcsaft_mcp_server.utils_mix.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_mix.mix_vp_feos")
    def test_mix_vp(self, mock_calc, mock_predict):
        """Test Mixture Vapor Pressure Logic"""
        mock_predict.side_effect = ["p1", "p2"]
        mock_calc.return_value = (101325.0, 90000.0)  # bubble and dew pressures

        params = utils_mix.MixVpParams(
            smiles_list=["A", "B"],
            mole_fractions=[0.5, 0.5],
            kij_matrix=[[0.0, 0.0], [0.0, 0.0]],
            min_temp=300,
            max_temp=310,
            npoints=5,
        )
        temps, bps, dps = utils_mix.mix_vp(params)

        self.assertEqual(len(temps), 5)
        self.assertEqual(len(bps), 5)
        self.assertEqual(len(dps), 5)

    @patch("gnnepcsaft_mcp_server.utils_mix.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_mix.critical_points_feos")
    @patch("gnnepcsaft_mcp_server.utils_mix.pure_vp_feos")
    @patch("gnnepcsaft_mcp_server.utils_mix.mix_vp_feos")
    def test_mix_vle_pxy(self, mock_mix_vp, mock_pure_vp, mock_critical, mock_predict):
        """Test Binary VLE P-x-y Diagram Logic"""
        mock_predict.side_effect = [[1, 2, 3], [1, 2, 3]]
        mock_critical.return_value = (400.0, 5e6, "unused")
        mock_pure_vp.return_value = 50000.0
        mock_mix_vp.return_value = (101325.0, 90000.0)

        res = utils_mix.mix_vle_pxy(
            ["A", "B"], [[0, 0], [0, 0]], temperature=350.0, npoints=5
        )

        # Result should be (x_values, bubble_pressures, dew_pressures)
        self.assertEqual(len(res), 3)
        xs, bps, dps = res
        self.assertGreater(len(xs), 0)
        self.assertEqual(len(bps), len(xs))
        self.assertEqual(len(dps), len(xs))

    @patch("gnnepcsaft_mcp_server.utils_mix.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_mix.mix_lle_diagram_feos")
    def test_mix_lle(self, mock_calc, mock_predict):
        """Test Mixture Liquid-Liquid Equilibrium Logic"""
        mock_predict.side_effect = [[1, 2, 3], [1, 2, 3]]
        expected_output = {
            "x0": [0.1, 0.2, 0.3],
            "x1": [0.9, 0.8, 0.7],
            "temperature": [300.0, 305.0, 310.0],
        }
        mock_calc.return_value = expected_output

        params = utils_mix.MixLLEParams(
            smiles_list=["A", "B"],
            mole_fractions=[0.5, 0.5],
            kij_matrix=[[0.0, 0.0], [0.0, 0.0]],
            temperature_min=300.0,
            temperature_max=310.0,
            pressure=101325.0,
            npoints=3,
        )
        res = utils_mix.mix_lle(params)

        self.assertEqual(res, expected_output)

    @patch("gnnepcsaft_mcp_server.utils_mix.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_mix.mix_vlle_diagram_feos")
    def test_mix_vlle(self, mock_calc, mock_predict):
        """Test Mixture Vapor-Liquid-Liquid Equilibrium Logic"""
        mock_predict.side_effect = [[1, 2, 3], [1, 2, 3]]
        vle_output = {"x0": [0.1], "y0": [0.9], "temperature": [300.0]}
        lle_output = {"x0": [0.2], "x1": [0.8], "temperature": [300.0]}
        vlle_output = {"x0": [0.15], "y0": [0.85], "x1": [0.75], "temperature": [300.0]}

        mock_calc.side_effect = [vle_output, lle_output, vlle_output]

        params = utils_mix.MixLLEParams(
            smiles_list=["A", "B"],
            mole_fractions=[0.5, 0.5],
            kij_matrix=[[0.0, 0.0], [0.0, 0.0]],
            temperature_min=300.0,
            temperature_max=310.0,
            pressure=101325.0,
            npoints=3,
        )
        res = utils_mix.mix_vlle(params)
        assert res is not None

        # Result should be a tuple of 3 dicts
        self.assertEqual(len(res), 3)

    @patch("gnnepcsaft_mcp_server.utils_mix.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_mix.critical_points_feos")
    @patch("gnnepcsaft_mcp_server.utils_mix.pure_vp_feos")
    @patch("gnnepcsaft_mcp_server.utils_mix.mix_vp_feos")
    def test_mix_ternary_vle_tx_fixed(
        self, mock_mix_vp, mock_pure_vp, mock_critical, mock_predict
    ):
        """Test Ternary VLE with fixed Temperature and Solvent Ratio"""
        mock_predict.side_effect = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        mock_critical.return_value = (400.0, 5e6, "unused")
        mock_pure_vp.return_value = 50000.0
        mock_mix_vp.return_value = (101325.0, 90000.0)

        params = utils_mix.TernaryVleTxParams(
            smiles_list=["A", "B", "C"],
            kij_matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            temperature=350.0,
            solvent_ratio=0.5,
            npoints=5,
        )
        xs, bps, dps = utils_mix.mix_ternary_vle_tx_fixed(params)

        self.assertGreater(len(xs), 0)
        self.assertEqual(len(bps), len(xs))
        self.assertEqual(len(dps), len(xs))

    @patch("gnnepcsaft_mcp_server.utils_mix.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_mix._get_ternary_lle_data")
    def test_mix_ternary_lle(self, mock_data, mock_predict):
        """Test Ternary Liquid-Liquid Equilibrium Logic"""
        mock_predict.side_effect = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        expected_output = {
            "x0": [0.1, 0.2],
            "x1": [0.3, 0.4],
            "x2": [0.6, 0.4],
            "y0": [0.5, 0.6],
            "y1": [0.3, 0.2],
            "y2": [0.2, 0.2],
        }
        mock_data.return_value = expected_output

        res = utils_mix.mix_ternary_lle(
            ["A", "B", "C"],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            temperature=350.0,
            pressure=101325.0,
            npoints=2,
        )

        self.assertEqual(res, expected_output)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    @patch("gnnepcsaft_mcp_server.utils_pure.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_pure.pure_den_feos")
    def test_pure_den_single_point(self, mock_calc, mock_predict):
        """Test pure density with single point"""
        mock_predict.return_value = "dummy_params"
        mock_calc.return_value = 1000.0

        temps, dens = utils_pure.pure_den("water", 300, 300, 101325, 1)

        self.assertEqual(len(temps), 1)
        self.assertEqual(len(dens), 1)

    @patch("gnnepcsaft_mcp_server.utils_mix.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_mix.mix_den_feos")
    def test_mix_den_different_pressures(self, mock_calc, mock_predict):
        """Test mixture density with varying pressures"""
        mock_predict.side_effect = ["p1", "p2"]
        mock_calc.return_value = 800.0

        params = utils_mix.MixDenParams(
            smiles_list=["C1", "C2"],
            mole_fractions=[0.3, 0.7],
            kij_matrix=[[0.0, 0.1], [0.1, 0.0]],
            min_temp=300,
            max_temp=350,
            pressure=500000,  # Higher pressure
            npoints=5,
        )
        temps, dens = utils_mix.mix_den(params)

        self.assertEqual(len(temps), 5)
        self.assertEqual(len(dens), 5)
        self.assertEqual(mock_predict.call_count, 2)

    @patch("gnnepcsaft_mcp_server.utils_pure.predict_pcsaft_parameters")
    @patch("gnnepcsaft_mcp_server.utils_pure.pure_h_lv_feos")
    def test_pure_h_lv_multiple_points(self, mock_calc, mock_predict):
        """Test enthalpy of vaporization with multiple points"""
        mock_predict.return_value = "dummy_params"
        mock_calc.side_effect = [40660.0, 40000.0, 39000.0]

        temps, h_lvs = utils_pure.pure_h_lv("water", 300, 320, 3)

        self.assertEqual(len(temps), 3)
        self.assertEqual(len(h_lvs), 3)
        self.assertEqual(h_lvs[0], 40660.0)
        self.assertEqual(h_lvs[1], 40000.0)
        self.assertEqual(h_lvs[2], 39000.0)


if __name__ == "__main__":
    unittest.main()
