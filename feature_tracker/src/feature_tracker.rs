use serde::{Deserialize, Serialize};
use image::{self, GrayImage, ImageBuffer, Luma, Pixel, imageops::*};

type ImageFloat = f32;

#[derive(Serialize, Deserialize, Debug)]
pub struct FeatureTrackingConfig {
    pub nlevels: usize,
    pub ratio: f64
}



pub struct FeatureTracker {
    config: FeatureTrackingConfig
}

fn build_image_pyramid(image: &GrayImage, nlevels: usize, ratio: f64) {
    let image: ImageBuffer<Luma<ImageFloat>, Vec<_>> = ImageBuffer::from_vec(
        image.width(), 
        image.height(),
        image.pixels()
            .map(|p| p.channels()[0] as ImageFloat)
            .collect::<Vec<_>>()
    ).unwrap();
    let (w0, h0) = image.dimensions();

    let mut pyramid = Vec::new();
    pyramid.push(image);
    for l in 0..nlevels {

        let level = i32::try_from(l).expect("Overflow caused by number of pyramid levels superior to i32::MAX. Consider decreasing number of pyramid levels");
        let nwidth = (f64::try_from(w0).unwrap() / ratio.powi(level)).round() as u32;
        let nheight = (f64::try_from(h0).unwrap() / ratio.powi(level)).round() as u32;
        
        let previous_image = pyramid.last().unwrap();
        pyramid.push(resize(previous_image, nwidth, nheight, Triangle));
    } 
}

impl FeatureTracker {
    pub fn new(config: FeatureTrackingConfig) -> Self {
        FeatureTracker { config }
    }

    pub fn process_frame<P: Pixel>(&self, image: &GrayImage) 
    {
        let pyramid = build_image_pyramid(image, self.config.nlevels, self.config.ratio);
        
    }
}


