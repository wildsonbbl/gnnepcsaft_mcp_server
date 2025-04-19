"unit tests for the utils module" ""

import json
from unittest.mock import MagicMock, patch

import pytest

from gnnepcsaft_mcp_server.utils import (
    batch_convert_pure_density_to_kg_per_m3,
    batch_critical_points,
    batch_inchi_to_smiles,
    batch_molecular_weights,
    batch_pa_to_bar,
    batch_predict_epcsaft_parameters,
    batch_pure_density,
    batch_pure_h_lv,
    batch_pure_vapor_pressure,
    batch_smiles_to_inchi,
    mixture_density,
    mixture_phase,
    mixture_vapor_pressure,
    predict_epcsaft_parameters,
    pubchem_description,
    pure_phase,
)

# Test data
METHANE_SMILES = "C"
ETHANOL_SMILES = "CCO"
WATER_SMILES = "O"
TEST_SMILES_LIST = [METHANE_SMILES, ETHANOL_SMILES, WATER_SMILES]


class TestPredictParameters:
    """Test parameter prediction functions"""

    def test_predict_epcsaft_parameters(self):
        """Test that parameter prediction returns expected format"""
        params = predict_epcsaft_parameters(METHANE_SMILES)
        assert isinstance(params, list)
        assert (
            len(params) == 8
        )  # [m, sigma, epsilon/kB, kappa_ab, epsilon_ab/kB, dipole moment, na, nb]
        assert all(isinstance(p, float) for p in params)

    def test_batch_predict_epcsaft_parameters(self):
        """Test batch parameter prediction"""
        params_list = batch_predict_epcsaft_parameters(TEST_SMILES_LIST)
        assert isinstance(params_list, list)
        assert len(params_list) == len(TEST_SMILES_LIST)
        assert all(len(params) == 8 for params in params_list)


class TestPhaseCalculations:
    """Test phase calculation functions"""

    def test_pure_phase_liquid(self):
        """Test pure phase determination for liquid"""
        phase = pure_phase(vapor_pressure=90000, system_pressure=100000)
        assert phase == "liquid"

    def test_pure_phase_vapor(self):
        """Test pure phase determination for vapor"""
        phase = pure_phase(vapor_pressure=110000, system_pressure=100000)
        assert phase == "vapor"

    def test_pure_phase_validation(self):
        """Test validation in pure_phase function"""
        with pytest.raises(AssertionError):
            pure_phase(-1, 100000)
        with pytest.raises(AssertionError):
            pure_phase(100000, -1)
        with pytest.raises(AssertionError):
            pure_phase("invalid", 100000)  # type: ignore

    def test_mixture_phase_liquid(self):
        """Test mixture phase determination for liquid"""
        phase = mixture_phase(
            bubble_point=90000, dew_point=120000, system_pressure=100000
        )
        assert phase == "liquid"

    def test_mixture_phase_vapor(self):
        """Test mixture phase determination for vapor"""
        phase = mixture_phase(
            bubble_point=110000, dew_point=120000, system_pressure=100000
        )
        assert phase == "vapor"

    def test_mixture_phase_two_phase(self):
        """Test mixture phase determination for two-phase"""
        phase = mixture_phase(
            bubble_point=110000, dew_point=90000, system_pressure=100000
        )
        assert phase == "two-phase"

    def test_mixture_phase_validation(self):
        """Test validation in mixture_phase function"""
        with pytest.raises(AssertionError):
            mixture_phase(-1, 100000, 100000)
        with pytest.raises(AssertionError):
            mixture_phase(100000, -1, 100000)
        with pytest.raises(AssertionError):
            mixture_phase(100000, 100000, -1)


class TestPubChemIntegration:
    """Test PubChem integration functions"""

    @patch("gnnepcsaft_mcp_server.utils.urlopen")
    def test_pubchem_description_success(self, mock_urlopen):
        """Test successful PubChem description retrieval"""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"Test": "Data"}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = pubchem_description(METHANE_SMILES)
        assert result == {"Test": "Data"}

    @patch("gnnepcsaft_mcp_server.utils.urlopen")
    def test_pubchem_description_error(self, mock_urlopen):
        """Test error handling in PubChem description retrieval"""
        mock_urlopen.side_effect = ValueError()

        result = pubchem_description(METHANE_SMILES)
        assert result == "no data available on this molecule in PubChem."


class TestMolecularProperties:
    """Test molecular properties calculation functions"""

    def test_batch_molecular_weights(self):
        """Test molecular weight calculation"""
        weights = batch_molecular_weights(TEST_SMILES_LIST)
        assert isinstance(weights, list)
        assert len(weights) == len(TEST_SMILES_LIST)
        assert all(isinstance(w, float) for w in weights)
        # Approximate check for methane molecular weight
        assert 15.5 < weights[0] < 16.5

    def test_batch_inchi_to_smiles(self):
        """Test InChI to SMILES conversion"""
        inchi_list = batch_smiles_to_inchi(TEST_SMILES_LIST)
        smiles_list = batch_inchi_to_smiles(inchi_list)

        # The conversion might not yield identical SMILES, but should be chemically equivalent
        assert len(smiles_list) == len(TEST_SMILES_LIST)
        assert all(isinstance(s, str) for s in smiles_list)

    def test_batch_smiles_to_inchi(self):
        """Test SMILES to InChI conversion"""
        inchi_list = batch_smiles_to_inchi(TEST_SMILES_LIST)
        assert isinstance(inchi_list, list)
        assert len(inchi_list) == len(TEST_SMILES_LIST)
        assert all(isinstance(i, str) for i in inchi_list)
        assert all(i.startswith("InChI=") for i in inchi_list)


class TestThermodynamicCalculations:
    """ "Test thermodynamic calculations functions"""

    def test_mixture_density(self):
        """Test mixture density calculation"""
        parameters = [
            [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.04],  # Methane-like
            [2.0, 3.5, 200.0, 0.01, 2000.0, 0.0, 1.0, 1.0, 46.07],  # Ethanol-like
        ]
        state = [298.15, 101325, 0.5, 0.5]  # T, P, x1, x2
        kij_matrix = [[0.0, 0.0], [0.0, 0.0]]

        density = mixture_density(parameters, state, kij_matrix)
        assert isinstance(density, float)
        assert density > 0

    def test_mixture_vapor_pressure(self):
        """Test mixture vapor pressure calculation"""
        parameters = [
            [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.04],  # Methane-like
            [2.0, 3.5, 200.0, 0.01, 2000.0, 0.0, 1.0, 1.0, 46.07],  # Ethanol-like
        ]
        state = [298.15, 0.0, 0.5, 0.5]  # T, P (not used), x1, x2
        kij_matrix = [[0.0, 0.0], [0.0, 0.0]]

        bubble_point, dew_point = mixture_vapor_pressure(parameters, state, kij_matrix)
        assert isinstance(bubble_point, float)
        assert isinstance(dew_point, float)
        assert bubble_point > 0
        assert dew_point > 0


class TestBatchCalculations:
    """Test batch calculations functions"""

    @patch("gnnepcsaft_mcp_server.utils.pure_den_feos")
    @patch("gnnepcsaft_mcp_server.utils.predict_epcsaft_parameters")
    def test_batch_pure_density(self, mock_predict, mock_den_feos):
        """Test batch pure density calculation"""
        mock_predict.return_value = [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_den_feos.return_value = 500.0

        state = [298.15, 101325]  # T, P
        densities = batch_pure_density(TEST_SMILES_LIST, state)

        assert isinstance(densities, list)
        assert len(densities) == len(TEST_SMILES_LIST)
        assert all(d == 500.0 for d in densities)

    @patch("gnnepcsaft_mcp_server.utils.pure_vp_feos")
    @patch("gnnepcsaft_mcp_server.utils.predict_epcsaft_parameters")
    def test_batch_pure_vapor_pressure(self, mock_predict, mock_vp_feos):
        """Test batch pure vapor pressure calculation"""
        mock_predict.return_value = [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_vp_feos.return_value = 50000.0

        temperature = 298.15
        pressures = batch_pure_vapor_pressure(TEST_SMILES_LIST, temperature)

        assert isinstance(pressures, list)
        assert len(pressures) == len(TEST_SMILES_LIST)
        assert all(p == 50000.0 for p in pressures)

    @patch("gnnepcsaft_mcp_server.utils.pure_h_lv_feos")
    @patch("gnnepcsaft_mcp_server.utils.predict_epcsaft_parameters")
    def test_batch_pure_h_lv(self, mock_predict, mock_h_lv_feos):
        """Test batch enthalpy of vaporization calculation"""
        mock_predict.return_value = [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_h_lv_feos.return_value = 20.0

        temperature = 298.15
        enthalpies = batch_pure_h_lv(TEST_SMILES_LIST, temperature)

        assert isinstance(enthalpies, list)
        assert len(enthalpies) == len(TEST_SMILES_LIST)
        assert all(h == 20.0 for h in enthalpies)

    @patch("gnnepcsaft_mcp_server.utils.critical_points_feos")
    @patch("gnnepcsaft_mcp_server.utils.predict_epcsaft_parameters")
    def test_batch_critical_points(self, mock_predict, mock_critical_points):
        """Test batch critical points calculation"""
        mock_predict.return_value = [1.0, 3.7, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_critical_points.return_value = [190.0, 4600000.0, 10200.0]

        critical_points = batch_critical_points(TEST_SMILES_LIST)

        assert isinstance(critical_points, list)
        assert len(critical_points) == len(TEST_SMILES_LIST)
        assert all(len(cp) == 3 for cp in critical_points)


class TestUnitConversions:
    """Test unit conversion functions"""

    def test_batch_pa_to_bar(self):
        """Test pressure conversion from Pa to bar"""
        pa_values = [100000.0, 200000.0, 300000.0]
        bar_values = batch_pa_to_bar(pa_values)

        assert isinstance(bar_values, list)
        assert len(bar_values) == len(pa_values)
        assert bar_values == [1.0, 2.0, 3.0]

    def test_batch_convert_pure_density_to_kg_per_m3(self):
        """Test density conversion from mol/m³ to kg/m³"""
        density_values = [1000.0, 2000.0, 3000.0]
        mw_values = [16.04, 46.07, 18.02]
        kg_per_m3_values = batch_convert_pure_density_to_kg_per_m3(
            density_values, mw_values
        )

        assert isinstance(kg_per_m3_values, list)
        assert len(kg_per_m3_values) == len(density_values)

        # Check calculations
        expected = [den * mw / 1000 for den, mw in zip(density_values, mw_values)]
        assert kg_per_m3_values == expected
