//! Persistent revival-only stream. Never reseed at a layer/unit/epoch boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevivalRng(u64);

impl Default for RevivalRng {
    fn default() -> Self {
        Self(0x9e3779b97f4a7c15)
    }
}

impl RevivalRng {
    pub fn signed_uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        2.0 * ((self.0 >> 40) as f32 / 16777216.0) - 1.0
    }

    // Checkpoint records are f32: four exact 16-bit limbs avoid u64 precision loss.
    pub fn encode(self) -> [f32; 4] {
        std::array::from_fn(|i| ((self.0 >> (16 * i)) & 0xffff) as f32)
    }

    pub fn decode(values: &[f32]) -> Result<Self, String> {
        if values.len() != 4 || values.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 65535.0 || v.fract() != 0.0) {
            return Err("invalid revival RNG checkpoint state (expected four u16 limbs)".into());
        }
        let state = values.iter().enumerate().fold(0u64, |s, (i, v)| s | ((*v as u64) << (16 * i)));
        if state == 0 {
            return Err("invalid zero revival RNG checkpoint state".into());
        }
        Ok(Self(state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn revival_rng_continues_exactly_after_serialization() {
        let mut rng = RevivalRng::default();
        let first: Vec<_> = (0..64).map(|_| rng.signed_uniform()).collect();
        let mut restored = RevivalRng::decode(&rng.encode()).unwrap();
        let second: Vec<_> = (0..64).map(|_| rng.signed_uniform()).collect();
        assert_ne!(first, second);
        assert_eq!(second, (0..64).map(|_| restored.signed_uniform()).collect::<Vec<_>>());
        let mut fresh = RevivalRng::default();
        assert_eq!(first, (0..64).map(|_| fresh.signed_uniform()).collect::<Vec<_>>());
    }
    #[test]
    fn revival_rng_validates_lossless_limbs() {
        let rng = RevivalRng(u64::MAX);
        assert_eq!(RevivalRng::decode(&rng.encode()).unwrap(), rng);
        for bad in [vec![], vec![0.0; 4], vec![1.5; 4], vec![65536.0; 4], vec![f32::NAN; 4], vec![-1.0; 4]] {
            assert!(RevivalRng::decode(&bad).is_err());
        }
    }
}
