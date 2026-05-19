import os

if os.environ.get('SKYLLH_DISABLE_RUST', '').strip().lower() in ('1', 'true', 'yes'):
    _rs = None
    RUST_AVAILABLE = False
else:
    try:
        import skyllh_rs as _rs

        RUST_AVAILABLE = True
    except ImportError:
        _rs = None
        RUST_AVAILABLE = False
