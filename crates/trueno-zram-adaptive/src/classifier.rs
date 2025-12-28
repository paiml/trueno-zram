//! Page classification for algorithm selection.

use trueno_zram_core::{Algorithm, PAGE_SIZE};

use crate::entropy::{EntropyCalculator, EntropyLevel};

/// Classifier that selects compression algorithm based on page characteristics.
#[derive(Debug, Default)]
pub struct PageClassifier {
    entropy_calc: EntropyCalculator,
}

impl PageClassifier {
    /// Create a new page classifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Classify a page and return the recommended algorithm.
    #[must_use]
    pub fn classify(&mut self, page: &[u8; PAGE_SIZE]) -> Algorithm {
        let level = self.entropy_calc.classify(page);

        match level {
            EntropyLevel::VeryLow => Algorithm::Lz4,
            EntropyLevel::Low => Algorithm::Lz4,
            EntropyLevel::Medium => Algorithm::Zstd { level: 3 },
            EntropyLevel::High => Algorithm::Zstd { level: 1 },
            EntropyLevel::VeryHigh => Algorithm::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_page_uses_lz4() {
        let mut classifier = PageClassifier::new();
        let page = [0u8; PAGE_SIZE];
        assert_eq!(classifier.classify(&page), Algorithm::Lz4);
    }

    #[test]
    fn test_random_page_skips_compression() {
        let mut classifier = PageClassifier::new();
        // Create high-entropy page
        let mut page = [0u8; PAGE_SIZE];
        let mut state = 12345u64;
        for byte in &mut page {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (state >> 33) as u8;
        }
        assert_eq!(classifier.classify(&page), Algorithm::None);
    }
}
