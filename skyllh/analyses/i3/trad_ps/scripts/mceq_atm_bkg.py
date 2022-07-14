import argparse
import numpy as np
import os.path
import pickle

import crflux.models as pm
import mceq_config as config
from MCEq.core import MCEqRun

from skyllh.analyses.i3.trad_ps.pd_aeff import PDAeff
from skyllh.datasets.i3 import PublicData_10y_ps

def create_flux_file(save_path, ds):
    """Creates a pickle file containing the flux for the given dataset.
    """
    output_filename = ds.get_aux_data_definition('mceq_flux_datafile')[0]
    output_pathfilename = ''
    if save_path is None:
        output_pathfilename = ds.get_abs_pathfilename_list([output_filename])[0]
    else:
        output_pathfilename = os.path.join(
            save_path, output_filename)

    print('Output path filename: %s'%(output_pathfilename))

    # Load the effective area instance to get the binning information.
    aeff = PDAeff(
        os.path.join(
            ds.root_dir,
            ds.get_aux_data_definition('eff_area_datafile')[0]
        )
    )

    # Setup MCeq.
    config.e_min = float(
        10**(np.max([aeff._log10_enu_binedges_lower[0], 2])))
    config.e_max = float(
        10**(np.min([aeff._log10_enu_binedges_upper[-1], 9])+0.05))

    print('E_min = %s'%(config.e_min))
    print('E_max = %s'%(config.e_max))

    mceq = MCEqRun(
        interaction_model="SIBYLL2.3c",
        primary_model=(pm.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
        density_model=("MSIS00_IC", ("SouthPole", "January")),
    )

    print('MCEq log10(e_grid) = %s'%(str(np.log10(mceq.e_grid))))

    mag = 0
    # Use the same binning as for the effective area.
    # theta = delta + pi/2
    print('sin_true_dec_binedges: %s'%(str(aeff.sin_decnu_binedges)))
    theta_angles_binedges = np.rad2deg(
        np.arcsin(aeff.sin_decnu_binedges) + np.pi/2
    )
    theta_angles = 0.5*(theta_angles_binedges[:-1] + theta_angles_binedges[1:])
    print('Theta angles = %s'%(str(theta_angles)))

    flux_def = dict()

    all_component_names = [
        "numu_conv",
        "numu_pr",
        "numu_total",
        "mu_conv",
        "mu_pr",
        "mu_total",
        "nue_conv",
        "nue_pr",
        "nue_total",
        "nutau_pr",
    ]

    # Initialize empty grid
    for frac in all_component_names:
        flux_def[frac] = np.zeros(
            (len(mceq.e_grid), len(theta_angles)))

    # fluxes calculated for different theta_angles
    for ti, theta in enumerate(theta_angles):
        mceq.set_theta_deg(theta)
        mceq.solve()

        # same meaning of prefixes for muon neutrinos as for muons
        flux_def["mu_conv"][:, ti] = (
            mceq.get_solution("conv_mu+", mag) +
            mceq.get_solution("conv_mu-", mag)
        )

        flux_def["mu_pr"][:, ti] = (
            mceq.get_solution("pr_mu+", mag) +
            mceq.get_solution("pr_mu-", mag)
        )

        flux_def["mu_total"][:, ti] = (
            mceq.get_solution("total_mu+", mag) +
            mceq.get_solution("total_mu-", mag)
        )

        # same meaning of prefixes for muon neutrinos as for muons
        flux_def["numu_conv"][:, ti] = (
            mceq.get_solution("conv_numu", mag) +
            mceq.get_solution("conv_antinumu", mag)
        )

        flux_def["numu_pr"][:, ti] = (
            mceq.get_solution("pr_numu", mag) +
            mceq.get_solution("pr_antinumu", mag)
        )

        flux_def["numu_total"][:, ti] = (
            mceq.get_solution("total_numu", mag) +
            mceq.get_solution("total_antinumu", mag)
        )

        # same meaning of prefixes for electron neutrinos as for muons
        flux_def["nue_conv"][:, ti] = (
            mceq.get_solution("conv_nue", mag) +
            mceq.get_solution("conv_antinue", mag)
        )

        flux_def["nue_pr"][:, ti] = (
            mceq.get_solution("pr_nue", mag) +
            mceq.get_solution("pr_antinue", mag)
        )

        flux_def["nue_total"][:, ti] = (
            mceq.get_solution("total_nue", mag) +
            mceq.get_solution("total_antinue", mag)
        )

        # since there are no conventional tau neutrinos, prompt=total
        flux_def["nutau_pr"][:, ti] = (
            mceq.get_solution("total_nutau", mag) +
            mceq.get_solution("total_antinutau", mag)
        )
    print("\U0001F973")

    # Save the result to the output file.
    with open(output_pathfilename, 'wb') as f:
        pickle.dump(((mceq.e_grid, theta_angles_binedges), flux_def), f)
    print('Saved fluxes for dataset %s to: %s'%(ds.name, output_pathfilename))

#-------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate atmospheric background fluxes with MCEq.'
    )
    parser.add_argument(
        '-b',
        '--data-base-path',
        type=str,
        default='/data/ana/analyses',
        help='The base path of the data repository.'
    )
    parser.add_argument(
        '-s',
        '--save-path',
        type=str,
        default=None
    )

    args = parser.parse_args()

    dsc = PublicData_10y_ps.create_dataset_collection(args.data_base_path)

    dataset_names = ['IC40', 'IC59', 'IC79', 'IC86_I', 'IC86_II']
    for ds_name in dataset_names:
        ds = dsc.get_dataset(ds_name)
        create_flux_file(
            save_path = args.save_path,
            ds=ds
        )
