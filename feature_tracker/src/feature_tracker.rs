use image::{self, ImageBuffer, Luma};

type ImageFloat = f32;
struct FeatureTracker<const NLEVELS: usize> {
    pyramid: [ImageBuffer<Luma<ImageFloat>, Vec<ImageFloat>>; NLEVELS]
}