use crate::types::*;


pub use pyramid_module::*; // Re-export to higher module
mod pyramid_module {
    use super::*;
    use image::{self, GrayImage, ImageBuffer, Luma, Pixel, imageops::*};
    pub trait ImagePyramidInput {
        fn width(&self) -> u32;
        fn height(&self) -> u32;
    
        /// Return the pixels as intensities between 0 and 1
        fn pixel_intensities(&self) -> Vec<Float>;
    }
    impl ImagePyramidInput for GrayImage {
        fn width(&self) -> u32 {
            self.width()
        }
        fn height(&self) -> u32 {
            self.height()
        }
        fn pixel_intensities(&self) -> Vec<Float> {
            self.pixels().map(|p| (p.channels()[0] as Float) / 255.0).collect()
        }
    }
    
    // Build pyramid
    pub fn build_image_pyramid<InImage: ImagePyramidInput>(
        in_image: &InImage, nlevels: usize, ratio: f64, bluring: bool, blur_sigma: Float
    ) -> Pyramid {
        let float_image: ImageBuffer<Luma<Float>, Vec<_>> = ImageBuffer::from_vec(
            in_image.width(), 
            in_image.height(),
            in_image.pixel_intensities()
        ).unwrap();
    
        let float_image = if bluring {
            blur(&float_image, blur_sigma.into())
        } else {
            float_image
        };
    
        let (w0, h0) = float_image.dimensions();
    
        let mut pyramid = Vec::new();
        pyramid.push(float_image);
        for l in 1..nlevels {
    
            let level = i32::try_from(l).expect("Overflow caused by number of pyramid levels superior to i32::MAX. Consider decreasing number of pyramid levels");
            let nwidth = (f64::try_from(w0).unwrap() / ratio.powi(level)).round() as u32;
            let nheight = (f64::try_from(h0).unwrap() / ratio.powi(level)).round() as u32;
            
            let previous_image = pyramid.last().unwrap();
            pyramid.push(resize(previous_image, nwidth, nheight, Triangle));
        } 
    
        pyramid
    
    }
}



pub use interpolation_module::*; // Re-export to higher module
mod interpolation_module {
    use super::*;
    use image::{self, GrayImage, ImageBuffer, Luma, Pixel, imageops::*};
    pub trait InterpolationPoint {
        fn as_array(&self) -> &[Float; 2];
    }
    impl InterpolationPoint for [Float; 2] {
        fn as_array(&self) -> &[Float; 2] {
            self
        }
    }
    impl InterpolationPoint for Vec2 {
        fn as_array(&self) -> &[Float; 2] {
            self.as_ref()
        }
    }
    
    
    // Bicubic interpolation
    pub fn interpolate_bicubic<PointT: InterpolationPoint>(point: &PointT, interpolant: FloatGreyImage)
    {   
        let [x, y] = point.as_array();
        let (w,h) = interpolant.dimensions();
    
    
    }
    
    
    #[cfg(test)]
    mod test {
    
        use super::*;
    
        #[test]
        fn test_interpolate_bicubic() {
            let p = [1.0f32, 2.0f32];
            let interpolant = FloatGreyImage::new(100, 100);
            interpolate_bicubic(&p, interpolant);
    
            let p = Vec2::new(1.0, 2.0);
            let interpolant = FloatGreyImage::new(100, 100);
            interpolate_bicubic(&p, interpolant);
        }
    }
}
