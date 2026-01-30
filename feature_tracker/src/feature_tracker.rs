use serde::{Deserialize, Serialize};
use image::{self, GrayImage, ImageBuffer, Luma, Pixel, imageops::*};

type ImageFloat = f32;
type Pyramid = Vec<ImageBuffer<Luma<ImageFloat>, Vec<ImageFloat>>>;

#[derive(Serialize, Deserialize, Debug)]
pub struct FeatureTrackingConfig {
    pub nlevels: usize,
    pub ratio: f64
}



pub struct FeatureTracker {
    config: FeatureTrackingConfig,
    previous_pyramid: Option<Pyramid>
}

impl FeatureTracker {
    pub fn new(config: FeatureTrackingConfig) -> Self {
        FeatureTracker { config, previous_pyramid: None }
    }

    pub fn process_frame(&mut self, image: &GrayImage) 
    {
        let pyramid = build_image_pyramid(image, self.config.nlevels, self.config.ratio);
        if let Some(previous_pyramid) = self.previous_pyramid.as_ref() {
            println!("there was a previous pyramid");
        };

        self.previous_pyramid = Some(pyramid);
    }

    pub fn get_pyramid(&self) -> Option<&Pyramid> {
        self.previous_pyramid.as_ref()
    }
}


fn build_image_pyramid(image: &GrayImage, nlevels: usize, ratio: f64) -> Pyramid {
    let image: ImageBuffer<Luma<ImageFloat>, Vec<_>> = ImageBuffer::from_vec(
        image.width(), 
        image.height(),
        image.pixels()
            .map(|p| (p.channels()[0] as ImageFloat) / 255.0)
            .collect::<Vec<_>>()
    ).unwrap();
    let (w0, h0) = image.dimensions();

    let mut pyramid = Vec::new();
    pyramid.push(image);
    for l in 1..nlevels {

        let level = i32::try_from(l).expect("Overflow caused by number of pyramid levels superior to i32::MAX. Consider decreasing number of pyramid levels");
        let nwidth = (f64::try_from(w0).unwrap() / ratio.powi(level)).round() as u32;
        let nheight = (f64::try_from(h0).unwrap() / ratio.powi(level)).round() as u32;
        
        let previous_image = pyramid.last().unwrap();
        pyramid.push(resize(previous_image, nwidth, nheight, Triangle));
    } 
    pyramid
}



#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_image_pyramid() {
        let (w, h) = (120, 60);
        let gray_image = GrayImage::from_vec(w, h, 
            Vec::from_iter(std::iter::repeat(5).take((w*h) as usize))).unwrap();

        let pyramid = build_image_pyramid(&gray_image, 3, 2.0);
        assert!(pyramid[0].dimensions() == (120, 60));
        assert!(pyramid[1].dimensions() == (60, 30));
        assert!(pyramid[2].dimensions() == (30, 15));
    }
}
