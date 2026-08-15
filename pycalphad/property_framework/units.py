import pint
import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING, Sequence
from pycalphad.core.composition_set import CompositionSet
if TYPE_CHECKING:
    from pycalphad.property_framework import ComputableProperty

ureg = pint.UnitRegistry(
    preprocessors=[lambda s: s.replace('%', ' percent ')],
    # Suppress warnings when redefining 'percent', '%', and 'ppm' units.
    # We intentionally redefine these relative to our custom 'fraction' unit.
    on_redefinition='ignore',
)
ureg.define('atom = 1/avogadro_number * mol')
ureg.define('fraction = []')
ureg.define('percent = 1e-2 fraction = %')
ureg.define('ppm = 1e-6 fraction')
pint.set_application_registry(ureg)
Q_ = ureg.Quantity
DimensionalityError = pint.DimensionalityError

def as_quantity(prop: "ComputableProperty", qt: npt.ArrayLike):
    if not isinstance(qt, Q_):
        return Q_(qt, prop.display_units)
    else:
        return qt

energy_implementation_units = GM_implementation_units = 'J / mol'
energy_display_units = GM_display_units = 'J / mol'
energy_display_name = GM_display_name = 'Gibbs Energy'
G_implementation_units = 'J'
G_display_units = 'J'
G_display_name = 'Gibbs Energy'
enthalpy_implementation_units = HM_implementation_units = GM_implementation_units
enthalpy_display_units = HM_display_units = GM_display_units
enthalpy_display_name = HM_display_name = 'Enthalpy'
H_implementation_units = 'J'
H_display_units = 'J'
H_display_name = 'Enthalpy'
entropy_implementation_units = SM_implementation_units = 'J / mol / K'
entropy_display_units = SM_display_units = 'J / mol / K'
entropy_display_name = SM_display_name = 'Entropy'
cpm_implementation_units = CPM_implementation_units = 'J / mol / K'
cpm_display_units = CPM_display_units = 'J / mol / K'
cpm_display_name = CPM_display_name = 'Heat Capacity'

# Molar volume and thermal expansion
VM_implementation_units = molar_volume_implementation_units = 'm**3 / mol'
VM_display_units = molar_volume_display_units = 'cm**3 / mol'
VM_display_name = molar_volume_display_name = 'Molar Volume'

V0_implementation_units = 'm**3 / mol'
V0_display_units = 'cm**3 / mol'
V0_display_name = 'Reference Molar Volume'

CTE_implementation_units = thermal_expansion_implementation_units = '1 / K'
CTE_display_units = thermal_expansion_display_units = '1e-6 / K'
CTE_display_name = thermal_expansion_display_name = 'Coefficient of Thermal Expansion'

# Elastic moduli & mechanical properties
youngs_modulus_implementation_units = E_implementation_units = 'Pa'
youngs_modulus_display_units = E_display_units = 'GPa'
youngs_modulus_display_name = E_display_name = "Young's Modulus"

shear_modulus_implementation_units = G_modulus_implementation_units = 'Pa'
shear_modulus_display_units = G_modulus_display_units = 'GPa'
shear_modulus_display_name = G_modulus_display_name = 'Shear Modulus'

bulk_modulus_implementation_units = K_modulus_implementation_units = 'Pa'
bulk_modulus_display_units = K_modulus_display_units = 'GPa'
bulk_modulus_display_name = K_modulus_display_name = 'Bulk Modulus'

poissons_ratio_implementation_units = nu_implementation_units = 'dimensionless'
poissons_ratio_display_units = nu_display_units = 'dimensionless'
poissons_ratio_display_name = nu_display_name = "Poisson's Ratio"

# Transport & interface properties
surface_tension_implementation_units = SIGMA_implementation_units = 'N / m'
surface_tension_display_units = SIGMA_display_units = 'mN / m'
surface_tension_display_name = SIGMA_display_name = 'Surface Tension'

viscosity_implementation_units = VISC_implementation_units = 'Pa * s'
viscosity_display_units = VISC_display_units = 'mPa * s'
viscosity_display_name = VISC_display_name = 'Dynamic Viscosity'

thermal_conductivity_implementation_units = THCD_implementation_units = 'W / m / K'
thermal_conductivity_display_units = THCD_display_units = 'W / m / K'
thermal_conductivity_display_name = THCD_display_name = 'Thermal Conductivity'

electrical_conductivity_implementation_units = 'S / m'
electrical_conductivity_display_units = 'MS / m'
electrical_conductivity_display_name = 'Electrical Conductivity'

electrical_resistivity_implementation_units = ELRS_implementation_units = 'ohm * m'
electrical_resistivity_display_units = ELRS_display_units = 'microohm * cm'
electrical_resistivity_display_name = ELRS_display_name = 'Electrical Resistivity'

def _conversions_per_formula_unit(compset):
    components = compset.phase_record.nonvacant_elements
    num_components = len(components)
    moles_per_fu = np.zeros((num_components,1))
    for comp_idx in range(num_components):
        compset.phase_record.formulamole_obj(moles_per_fu[comp_idx, :], compset.dof, comp_idx)
    # now we have 'moles per formula unit'
    # need to convert by adding molecular weight of each element
    grams_per_mol = np.array(compset.phase_record.molar_masses, dtype='float')
    grams_per_fu = np.dot(grams_per_mol, moles_per_fu)
    return moles_per_fu.sum(), grams_per_fu

def unit_conversion_context(compsets, prop):
    context = pint.Context()
    # these will be something/mol by convention
    # XXX: This is a very rough check
    if not ('/ mol' in str(prop.implementation_units)):
        return context
    implementation_units = (ureg.Unit(prop.implementation_units) * ureg.Unit('mol'))
    molar_weight = 0.0 # g/mol-atom
    for compset in compsets:
        if compset.NP > 0:
            moles_per_fu, grams_per_fu = _conversions_per_formula_unit(compset)
            grams_per_mol_atoms = (compset.NP / moles_per_fu) * grams_per_fu
            molar_weight += grams_per_mol_atoms
    molar_weight = Q_(molar_weight, 'g/mol')
    per_moles = ureg.get_dimensionality(ureg.Unit('{} / mol'.format(implementation_units)))
    per_mass = ureg.get_dimensionality(ureg.Unit('{} / g'.format(implementation_units)))

    context.add_transformation(
        per_moles,
        per_mass,
        lambda ureg, x: np.true_divide(x, molar_weight).to_reduced_units(),
    )
    context.add_transformation(
        per_mass,
        per_moles,
        lambda ureg, x: (x * molar_weight).to_reduced_units()
    )

    return context


def _composition_sets_for_unit_conversion(comp_sets: Sequence[CompositionSet], prop: "ComputableProperty") -> Sequence[CompositionSet]:
    """Select composition sets matching a phase-specific property for unit conversion context."""
    phase_name = getattr(prop, 'phase_name', None)
    if phase_name is None or phase_name == '*':
        return comp_sets

    tokens = phase_name.split('#')
    phase_name = tokens[0]
    multiplicity = int(tokens[1]) if len(tokens) > 1 else 1
    multiplicity_seen = 0
    for comp_set in comp_sets:
        if comp_set.phase_record.phase_name == phase_name:
            multiplicity_seen += 1
            if multiplicity_seen == multiplicity:
                return [comp_set]
    return comp_sets


def to_display_units(value: npt.ArrayLike, comp_sets: Sequence[CompositionSet], prop: "ComputableProperty") -> npt.ArrayLike:
    """Convert a property value from implementation units to display units.

    Filters composition sets to match phase-specific properties before
    building the molar mass context for per-mole to per-mass conversions.
    """
    implementation_units = ureg.Unit(getattr(prop, 'implementation_units', ''))
    display_units = ureg.Unit(getattr(prop, 'display_units', ''))
    context = unit_conversion_context(_composition_sets_for_unit_conversion(comp_sets, prop), prop)
    return Q_(value, implementation_units).to(display_units, context).magnitude