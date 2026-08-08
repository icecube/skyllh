use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Rust port of the argsort + searchsorted core of RandomChoice.__call__().
///
/// Given a pre-drawn uniform_values array (from rss.random.random(size))
/// and the pre-built cdf array, returns the selected indices into self._items.
/// The numpy RNG call stays in Python so seed reproducibility is preserved.
#[pyfunction]
pub fn weighted_choice_indices<'py>(
    py: Python<'py>,
    cdf: PyReadonlyArray1<f64>,
    uniform_values: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    let cdf_sl: Vec<f64> = cdf.as_array().iter().copied().collect();
    let uv_sl: Vec<f64> = uniform_values.as_array().iter().copied().collect();
    let n = uv_sl.len();
    let m = cdf_sl.len();

    // Sort positions by ascending uniform value (matches np.argsort).
    let mut idxs_of_sort: Vec<usize> = (0..n).collect();
    idxs_of_sort.sort_unstable_by(|&a, &b| {
        uv_sl[a].partial_cmp(&uv_sl[b]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Two-pointer scan over sorted (uniform_values, cdf): O(n + m) vs O(n log m)
    // binary search. Both sequences are ascending, so cdf_pos only advances.
    // Equivalent to np.searchsorted(cdf, u, side='right') for each u.
    let mut idxs = vec![0usize; n];
    let mut cdf_pos = 0usize;
    for &orig_pos in idxs_of_sort.iter() {
        let u = uv_sl[orig_pos];
        while cdf_pos < m && cdf_sl[cdf_pos] <= u {
            cdf_pos += 1;
        }
        idxs[orig_pos] = cdf_pos;
    }

    Ok(PyArray1::from_vec_bound(py, idxs))
}
