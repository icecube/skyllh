use pyo3::prelude::*;

mod llhratio;
mod random_choice;

#[pymodule]
fn skyllh_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(llhratio::log_lambda_and_grads, m)?)?;
    m.add_function(wrap_pyfunction!(random_choice::weighted_choice_indices, m)?)?;
    Ok(())
}
