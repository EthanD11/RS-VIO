use crate::types::*;


pub use pyramid_module::*; // Re-export to higher module
mod pyramid_module {
    use super::*;
    use image::{self, GrayImage, ImageBuffer, Luma, imageops::*};
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
            // self.pixels().map(|p| (p.channels()[0] as Float)).collect()
            self.as_raw().into_iter().map(|&p| p as Float).collect()
        }
    }
    impl ImagePyramidInput for FloatGrayImage {
        fn width(&self) -> u32 {
            self.width()
        }
        fn height(&self) -> u32 {
            self.height()
        }
        fn pixel_intensities(&self) -> Vec<Float> {
            self.as_raw().clone()
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

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn test_image_pyramid() {
            let (w, h) = (120, 60);
            let gray_image = GrayImage::from_vec(w, h, 
                Vec::from_iter(std::iter::repeat(5).take((w*h) as usize))).unwrap();

            let pyramid = build_image_pyramid(&gray_image, 3, 2.0, false, 0.0);
            assert!(pyramid[0].dimensions() == (120, 60));
            assert!(pyramid[1].dimensions() == (60, 30));
            assert!(pyramid[2].dimensions() == (30, 15));
        }
    }
}



pub use interpolation_module::*; // Re-export to higher module
mod interpolation_module {
    use std::ops::Not;

    use super::*;
    use nalgebra as na;
    use image::{self, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel, imageops::*};

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

    pub trait Interpolant {
        fn dimensions(&self) -> (u32, u32);
        fn intensity_at(&self, x: u32, y: u32) -> Float;
    }
    impl Interpolant for FloatGrayImage {
        fn dimensions(&self) -> (u32, u32) {
            self.dimensions()
        }
        fn intensity_at(&self, x: u32, y: u32) -> Float {
            // self.get_pixel(x, y).channels()[0]
            self.as_raw()[(y*self.width() + x) as usize]
        }
    }
    
    
    /// Performs bicubic interpolation of the interpolant.
    /// Returns `None` if the pixel is out-of-bound
    pub fn interpolate_bicubic<PointT: InterpolationPoint, InterpolantT: Interpolant>(
        point: &PointT, interpolant: &InterpolantT
    ) -> Option<Float>
    {   
        let &[x, y] = point.as_array();
        let (w,h) = interpolant.dimensions();

        let xf = x.floor() as u32;
        let yf = y.floor() as u32;

        if  (1..=w.saturating_sub(3)).contains(&xf).not() || 
            (1..=h.saturating_sub(3)).contains(&yf).not() 
        {
            return None // Out-of-bound
        }
        
        
        let x_low = xf - 1;
        let y_low = yf - 1;
        let tx = x - (xf as Float);
        let ty = y - (yf as Float);
        let mut fy = [0.0; 4];
        for dy in 0..4 {
            let v = y_low+dy;
            let fx = [
                interpolant.intensity_at(x_low, v),
                interpolant.intensity_at(x_low+1, v),
                interpolant.intensity_at(x_low+2, v),
                interpolant.intensity_at(x_low+3, v),
            ];
            fy[dy as usize] = bicubic_1d(&fx, tx);
        }

        let result = bicubic_1d(&fy, ty);

        Some(result)
    }


    /// Performs bicubic interpolation of the interpolant.
    /// Returns `None` if the pixel is out-of-bound
    pub fn d_interpolate_bicubic<PointT: InterpolationPoint, InterpolantT: Interpolant>(
        point: &PointT, 
        interpolant: &InterpolantT,
        dinterpolant_dpoint: &mut na::Vector2<Float>
    ) -> Option<Float>
    {   
        let &[x, y] = point.as_array();
        let (w,h) = interpolant.dimensions();

        let xf = x.floor() as u32;
        let yf = y.floor() as u32;

        if  (1..=w.saturating_sub(3)).contains(&xf).not() || 
            (1..=h.saturating_sub(3)).contains(&yf).not() 
        {
            return None // Out-of-bound
        }
        
        
        let x_low = xf - 1;
        let y_low = yf - 1;
        let tx = x - (xf as Float);
        let ty = y - (yf as Float);
        let mut fy = [0.0; 4];
        let mut dfy_tx = [0.0; 4];
        for dy in 0..4 {
            let v = y_low+dy;
            let fx = [
                interpolant.intensity_at(x_low, v),
                interpolant.intensity_at(x_low+1, v),
                interpolant.intensity_at(x_low+2, v),
                interpolant.intensity_at(x_low+3, v),
            ];

            let idx = dy as usize;
            fy[idx] = d_bicubic_1d(&fx, tx, &mut dfy_tx[idx]);
        }

        let mut dout_dty = 0.0;
        let mut dout_dfy = [0.0; 4];
        let out = d_bicubic_1d_full(fy, &mut dout_dfy, ty, &mut dout_dty);

        let dout_dtx = dout_dfy[0]*dfy_tx[0] + dout_dfy[1]*dfy_tx[1] + dout_dfy[2]*dfy_tx[2] + dout_dfy[3]*dfy_tx[3] ;

        dinterpolant_dpoint[0] = dout_dtx;
        dinterpolant_dpoint[1] = dout_dty;

        Some(out)
    }

    #[inline]
    fn bicubic_1d(
        f: &[Float; 4], t: Float
    ) -> Float
    {
        let a0 = f[1];
        let a1_double = f[2] - f[0];
        let a2_double = 2.0*f[0] - 5.0*f[1] + 4.0*f[2] - f[3];
        let a3_double = 3.0*(f[1]-f[2]) + f[3] - f[0];
        let result = a0 + 0.5 * (t * ( a1_double + t * ( a2_double + t * a3_double )));
        result
    }

    #[inline]
    fn d_bicubic_1d(
        f: &[Float; 4],
        t: Float, 
        dout_dt: &mut Float
    ) -> Float
    {   
        let a0 = f[1];
        let a1_double = f[2] - f[0];
        let a2_double = 2.0*f[0] - 5.0*f[1] + 4.0*f[2] - f[3];
        let a3_double = 3.0*(f[1]-f[2]) + f[3] - f[0];
        let result = a0 + 0.5 * (t * ( a1_double + t * ( a2_double + t * a3_double )));

        *dout_dt =  0.5 * (( a1_double + t * (2.0*a2_double + t * 3.0*a3_double )));

        result
    }

    fn d_bicubic_1d_full(
        f: [Float; 4], 
        dout_df: &mut [Float; 4],
        t: Float, 
        dout_dt: &mut Float,
    ) -> Float
    {   
        let a0 = f[1];
        let a1_double = f[2] - f[0];
        let a2_double = 2.0*f[0] - 5.0*f[1] + 4.0*f[2] - f[3];
        let a3_double = 3.0*(f[1]-f[2]) + f[3] - f[0];
        let result = a0 + 0.5 * (t * ( a1_double + t * ( a2_double + t * a3_double )));

        *dout_dt =  0.5 * (( a1_double + t * (2.0*a2_double + t * 3.0*a3_double )));
        dout_df[0] = 0.5 * (t * ( -1.0 + t * ( 2.0 + t * -1.0 )));
        dout_df[1] = 1.0 + 0.5 * (t * ( t * ( -5.0 + t * 3.0 )));
        dout_df[2] = 0.5 * (t * ( 1.0 + t * ( 4.0 + t * -3.0 )));
        dout_df[3] = 0.5 * (t * (t * ( -1.0 + t )));

        result
    }
    
    
    #[cfg(test)]
    mod test {
    
        use super::*;
    
        #[test]
        fn test_interpolate_bicubic() {
            let p = [1.0f32, 2.0f32];
            let interpolant = FloatGrayImage::new(100, 100);
            interpolate_bicubic(&p, &interpolant);
    
            let p = Vec2::new(1.0, 2.0);
            let interpolant = FloatGrayImage::new(100, 100);
            interpolate_bicubic(&p, &interpolant);
        }


        use rand::{self, Rng};
        use rand_distr::StandardNormal;
        #[test]
        fn test_d_bicubic() {
            let delta = f32::EPSILON.sqrt(); // Theoretical best value for finite differences
            let tol = 50.0*delta;
            let ntests = 10;
            let mut rng = rand::rng();
            
            // Test d_t
            for _ in 0..ntests {
                let f: [f32; 4] = std::array::from_fn(|_i| (&mut rng).sample(StandardNormal)); //.expect("Array of 4 elements");
                let t: f32 = (&mut rng).sample(StandardNormal);
                
                let dt = delta; // Best theoretical value for precision
                
                let out_minus = bicubic_1d(&f, t - dt);
                let out_plus = bicubic_1d(&f, t + dt);
                let dout_t_fd = (out_plus - out_minus) / (2.0*delta); // Finite differencing
                
                let mut dout_t = 0.0;
                d_bicubic_1d(&f, t, &mut dout_t);

                let mut dout_t_full = 0.0;
                d_bicubic_1d_full(f, &mut [0.0; 4], t, &mut dout_t_full);

                let error = (dout_t_fd - dout_t).abs();
                assert!(error <= tol, "delta is bigger than tol: {error} > {tol}");

                let error = (dout_t_fd - dout_t_full).abs();
                assert!(error <= tol, "delta is bigger than tol: {error} > {tol}");

                for i in 0..4 {
                    let f: [f32; 4] = std::array::from_fn(|_i| (&mut rng).sample(StandardNormal)); //.expect("Array of 4 elements");
                    let t: f32 = (&mut rng).sample(StandardNormal);

                    let mut df = [0.0; 4];
                    df[i] = delta;

                    let fplus = std::array::from_fn(|i| f[i] + df[i]);
                    let out_plus  = bicubic_1d(&fplus, t);
                    let fminus = std::array::from_fn(|i| f[i] - df[i]);
                    let out_minus = bicubic_1d(&fminus, t);
                    let dout_fi_fd = (out_plus - out_minus) / (2.0*delta);

                    let mut dout_t = 0f32;
                    let mut dout_f = [0f32; 4];
                    d_bicubic_1d_full(f, &mut dout_f, t, &mut dout_t);

                    let error = (dout_f[i] - dout_fi_fd).abs();
                    assert!(
                        error <= tol, 
                        "delta is bigger than tol: {error} > {tol}\n\
                        Finite difference: dout_f{i} = {dout_fi_fd}\n\
                        Tested grad method: dout_f{i} = {}",
                        dout_f[i]
                    );
                }


            }

            


        }

    }

    
}
