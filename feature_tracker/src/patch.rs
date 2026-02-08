
use crate::{image_operations::{d_interpolate_bicubic, interpolate_bicubic}, types::*};
use nalgebra::{self as na, ArrayStorage, Point2, U3, U52};

pub trait Patch<const SIZE: usize> {

    const PATTERN_RAW: [[f32; 2]; SIZE];

    /// Returns patch indexes as an array
    fn raw_indexes() -> [[f32; 2]; SIZE]; 

    fn shifted_indexes(&self, transform: &na::Isometry2<Float>) -> [[f32; 2]; SIZE];

    fn shifted_center(&self, transform: &na::Isometry2<Float>) -> [f32; 2];

    fn intensities(&self) -> [f32; SIZE];

    fn residuals(&self, img: &FloatGrayImage, transform: &na::Isometry2<Float>) -> na::SVector<Float, SIZE>;

    fn jacobian(&self) -> &na::SMatrix<Float, SIZE, 3>;

    fn inverse_hessian(&self) -> &na::Matrix3<Float>;
}

pub trait DensePatch<const SIZE: usize, const NROWS: usize>: Patch<SIZE> {
    const PATTERN_BY_ROWS: [[[i64; 2]; 2]; NROWS];

    /// Returns patch indexes as rows
    /// This assumes the patch can be described as follows:
    /// 
    /// Row i: `[[x_start_i, y_start_i], [x_end_i, y_end_i]]`
    /// 
    /// where all the pixels between start_i and end_i are part of the patch.
    fn rows() ->  [[[i64; 2]; 2]; NROWS];
}

pub struct Patch52<'a> {
    img: &'a FloatGrayImage,
    center_coords: [f32; 2],
    intensities: [f32; 52],
    dr_dtwist: na::SMatrix<Float, 52, 3>,
    inverse_hessian: na::Matrix3<Float>
}

use std::ops::Not;
impl<'a> Patch52<'a> {
    pub fn new(img: &'a FloatGrayImage, center: &[f32; 2], lm_lambda: Float) -> Patch52<'a>
    {
        let (intensities, dr_dtwist) = Self::compute_intensities_and_jacobian(img, *center);

        
        // hessian = lm_lambda*I + dr_dtwist.transpose() * dr_dtwist
        let mut hessian = na::Matrix3::from_diagonal_element(lm_lambda);
        hessian.gemm(1.0, &dr_dtwist.transpose(), &dr_dtwist, 1.0);
        if hessian.try_inverse_mut().not() {
            panic!("Hessian should be invertible by its definition")
        };
        let inverse_hessian = hessian;

        Self { img, center_coords: *center, intensities, dr_dtwist, inverse_hessian}
    }

    fn compute_intensities_and_jacobian(
        img: &FloatGrayImage, center: [f32; 2]
    ) -> ([Float; 52], na::SMatrix<Float, 52, 3>)
    {

        let mut intensities = [0.0; 52];
        let mut dr_dtwist = na::SMatrix::<_, 52, _>::zeros(); 

        let mut dtransform_dtwist = na::Matrix2x3::new(
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        );

        
        let mut dimg_dpixel = RowVec2::zeros();
        for ((shift, mut dri_dtwist), intensity) in Self::raw_indexes().iter()
            .zip(dr_dtwist.row_iter_mut())
            .zip(intensities.iter_mut())
        {
            
            // ri(twist) = img(transform(twist)*pi) - img_target
            // transform(twist) = exp(twist)
            
            // dri_dtwist(0) = dimg_dpixel(pi) * dtransform_dtwist
            // dtransform_dtwist = [
            //   [ -p2  1  0 ],
            //   [  p1  0  1 ]
            //  ]
            
            let pixel_coords = [
                center[0]+shift[0], 
                center[1]+shift[1]
            ];

            dtransform_dtwist[(0,0)] = -pixel_coords[1];
            dtransform_dtwist[(1,0)] =  pixel_coords[0];

            *intensity = d_interpolate_bicubic(&pixel_coords, img, &mut dimg_dpixel)
                .unwrap_or(0.0);
                // .expect("Patch should be fully inbound of host image");

            // dri_dtwist = dimg_dpixel * dtransform_dtwist;
            dimg_dpixel.mul_to(&dtransform_dtwist, &mut dri_dtwist);

            
        }
        
        (intensities, dr_dtwist)
    }

}

impl<'a> Patch<52> for Patch52<'a> {
    const PATTERN_RAW: [[f32; 2]; 52] = [
        [-3.0, 7.0],  [-1.0, 7.0],  [1.0, 7.0],   [3.0, 7.0],

        [-5.0, 5.0],  [-3.0, 5.0],  [-1.0, 5.0],  [1.0, 5.0],   [3.0, 5.0],  [5.0, 5.0],

        [-7.0, 3.0],  [-5.0, 3.0],  [-3.0, 3.0],  [-1.0, 3.0],  [1.0, 3.0],  [3.0, 3.0],
        [5.0, 3.0],   [7.0, 3.0],

        [-7.0, 1.0],  [-5.0, 1.0],  [-3.0, 1.0],  [-1.0, 1.0],  [1.0, 1.0],  [3.0, 1.0],
        [5.0, 1.0],   [7.0, 1.0],

        [-7.0, -1.0], [-5.0, -1.0], [-3.0, -1.0], [-1.0, -1.0], [1.0, -1.0], [3.0, -1.0],
        [5.0, -1.0],  [7.0, -1.0],

        [-7.0, -3.0], [-5.0, -3.0], [-3.0, -3.0], [-1.0, -3.0], [1.0, -3.0], [3.0, -3.0],
        [5.0, -3.0],  [7.0, -3.0],

        [-5.0, -5.0], [-3.0, -5.0], [-1.0, -5.0], [1.0, -5.0],  [3.0, -5.0], [5.0, -5.0],

        [-3.0, -7.0], [-1.0, -7.0], [1.0, -7.0],  [3.0, -7.0]
    ];

    #[inline]
    fn raw_indexes() -> [[f32; 2]; 52] {
        Self::PATTERN_RAW.map(|coords| coords.map(|c| c / 2.0))
    }

    fn shifted_indexes(&self, transform: &na::Isometry2<Float>) -> [[f32; 2]; 52] {
        let center = self.center_coords;
        let mut indexes = [[0.0f32; 2]; 52];
        for (shift, index) in Self::raw_indexes().iter().zip(indexes.iter_mut())
        {
            let pixel_coords = Point2::new(
                center[0]+shift[0], 
                center[1]+shift[1]
            );
            
            let pixel_coords = transform * pixel_coords;
            index[0] = pixel_coords[0];
            index[1] = pixel_coords[1];
        }
        indexes
    }

    fn shifted_center(&self, transform: &nalgebra::Isometry2<Float>) -> [f32; 2] {        
        let center = self.center_coords;
        let pixel_coords = Point2::new(
            center[0], 
            center[1]
        );
        let pixel_coords = transform * pixel_coords;
        let mut index = [0.0f32; 2];
        index[0] = pixel_coords[0];
        index[1] = pixel_coords[1];
        index
    }

    fn intensities(&self) -> [f32; 52] {
        self.intensities
    }

    fn residuals(&self, img: &FloatGrayImage, transform: &na::Isometry2<Float>) -> nalgebra::SVector<Float, 52> {
        let center = self.center_coords;
        let mut residuals = na::SVector::<_, 52>::zeros();
        for ((shift, res), intensity_host) in Self::raw_indexes().iter()
            .zip(residuals.iter_mut())
            .zip(self.intensities.iter())
        {
            let pixel_coords = Point2::new(
                center[0]+shift[0], 
                center[1]+shift[1]
            );

            let pixel_coords = transform * pixel_coords;
            *res = interpolate_bicubic(&pixel_coords, img).unwrap_or(0.0) - intensity_host;
        }
        residuals
    }

    fn jacobian(&self) -> &nalgebra::SMatrix<Float, 52, 3> {
        &self.dr_dtwist
    }

    fn inverse_hessian(&self) -> &nalgebra::Matrix3<Float> {
        &self.inverse_hessian
    }
}



impl<'a> DensePatch<52, 8> for Patch52<'a> {
    const PATTERN_BY_ROWS: [[[i64; 2]; 2]; 8] = [
        [[-3, 7], [3, 7]],

        [[-5, 5], [5, 5]],

        [[-7, 3],  [7, 3]],

        [[-7, 1],  [7, 1]],

        [[-7, -1], [7, -1]],

        [[-7, -3], [7, -3]],

        [[-5, -5], [5, -5]],

        [[-3, -7], [3, -7]]
    ];

    #[inline]
    fn rows() ->  [[[i64; 2]; 2]; 8] {
        Self::PATTERN_BY_ROWS
    }
}