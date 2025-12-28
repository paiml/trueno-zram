//! CPU feature detection for SIMD backends.

use crate::SimdBackend;

/// Detected CPU SIMD features.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdFeatures {
    /// SSE4.2 support.
    pub sse42: bool,
    /// AVX2 support.
    pub avx2: bool,
    /// AVX-512F support.
    pub avx512f: bool,
    /// AVX-512BW support.
    pub avx512bw: bool,
    /// ARM NEON support.
    pub neon: bool,
}

impl SimdFeatures {
    /// Detect features on the current CPU.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse42: std::arch::is_x86_feature_detected!("sse4.2"),
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
                neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse42: false,
                avx2: false,
                avx512f: false,
                avx512bw: false,
                neon: true, // NEON is mandatory on AArch64
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    /// Check if AVX-512 is fully supported.
    #[must_use]
    pub fn has_avx512(&self) -> bool {
        self.avx512f && self.avx512bw
    }
}

/// Detect the best available SIMD backend.
#[must_use]
pub fn detect_backend() -> SimdBackend {
    let features = SimdFeatures::detect();

    if features.has_avx512() {
        SimdBackend::Avx512
    } else if features.avx2 {
        SimdBackend::Avx2
    } else if features.sse42 {
        SimdBackend::Sse42
    } else if features.neon {
        SimdBackend::Neon
    } else {
        SimdBackend::Scalar
    }
}

/// Check if a specific backend is available.
#[must_use]
pub fn is_available(backend: SimdBackend) -> bool {
    let features = SimdFeatures::detect();

    match backend {
        SimdBackend::Scalar => true,
        SimdBackend::Sse42 => features.sse42,
        SimdBackend::Avx2 => features.avx2,
        SimdBackend::Avx512 => features.has_avx512(),
        SimdBackend::Neon => features.neon,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_returns_valid_features() {
        let features = SimdFeatures::detect();
        // On any platform, this should not panic
        let _ = features.sse42;
        let _ = features.avx2;
        let _ = features.neon;
    }

    #[test]
    fn test_scalar_always_available() {
        assert!(is_available(SimdBackend::Scalar));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_has_sse42_or_better() {
        // All x86_64 CPUs since 2008 have SSE4.2
        let features = SimdFeatures::detect();
        assert!(features.sse42);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64_has_neon() {
        let features = SimdFeatures::detect();
        assert!(features.neon);
    }

    #[test]
    fn test_detect_backend_consistency() {
        let backend = detect_backend();
        assert!(is_available(backend));
    }
}
