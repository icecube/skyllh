use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// Must match ZeroSigH0SingleDatasetTCLLHRatio._one_plus_alpha = 1e-3
const ONE_PLUS_ALPHA: f64 = 1e-3;
const ALPHA: f64 = ONE_PLUS_ALPHA - 1.0; // -0.999

/// Rust port of ZeroSigH0SingleDatasetTCLLHRatio.calculate_log_lambda_and_grads().
///
/// Returns (log_lambda, grads, nsgrad_i) where grads is the full gradient
/// vector and nsgrad_i is returned so the caller can populate _cache_nsgrad_i.
#[pyfunction]
pub fn log_lambda_and_grads<'py>(
    py: Python<'py>,
    n: usize,
    ns: f64,
    ns_pidx: usize,
    p_mask: PyReadonlyArray1<bool>,
    xi: PyReadonlyArray1<f64>,
    dxi_dp: PyReadonlyArray2<f64>,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let xi_arr = xi.as_array();
    let dxi_dp_arr = dxi_dp.as_array();
    let p_mask_arr = p_mask.as_array();

    let n_selected = xi_arr.len();
    let n_pure_bkg = n - n_selected;
    let n_params = p_mask_arr.len();
    let n_non_ns = dxi_dp_arr.ncols();

    let mut ll_sum = 0.0f64;
    let mut ng_sum = 0.0f64;
    let mut nsgrad_i = vec![0.0f64; n_selected];
    let mut p_grads = vec![0.0f64; n_non_ns];

    // Single pass: compute log_lambda, nsgrad_i, and p_grads together so each
    // event's data is touched only once and no intermediate Vec is allocated.
    for i in 0..n_selected {
        let xv = xi_arr[i];
        let ai = ns * xv;
        let (ll_i, ng_i, factor) = if ai > ALPHA {
            let oo1pai = 1.0 / (1.0 + ai);
            (ai.ln_1p(), xv * oo1pai, ns * oo1pai)
        } else {
            // Taylor expansion to avoid catastrophic cancellation near -1.
            let tilde = (ai - ALPHA) / ONE_PLUS_ALPHA;
            let ng = (1.0 - tilde) * xv / ONE_PLUS_ALPHA;
            let f = ns * (1.0 - tilde) / ONE_PLUS_ALPHA;
            (ALPHA.ln_1p() + tilde - 0.5 * tilde * tilde, ng, f)
        };

        ll_sum += ll_i;
        ng_sum += ng_i;
        nsgrad_i[i] = ng_i;

        let row = dxi_dp_arr.row(i);
        for (j, &d) in row.iter().enumerate() {
            p_grads[j] += factor * d;
        }
    }

    let log_lambda = ll_sum + (n_pure_bkg as f64) * (-ns / n as f64).ln_1p();

    // Build grads vector.
    let mut grads = vec![0.0f64; n_params];
    grads[ns_pidx] = ng_sum - (n_pure_bkg as f64) / (n as f64 - ns);

    // Map p_grads into the full grads vector via p_mask.
    let mut p_idx = 0usize;
    for (i, &in_mask) in p_mask_arr.iter().enumerate() {
        if in_mask {
            grads[i] = p_grads[p_idx];
            p_idx += 1;
        }
    }

    Ok((
        log_lambda,
        PyArray1::from_vec_bound(py, grads),
        PyArray1::from_vec_bound(py, nsgrad_i),
    ))
}
