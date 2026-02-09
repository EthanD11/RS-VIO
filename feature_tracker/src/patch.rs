
use crate::{image_operations::{d_interpolate_bicubic, interpolate_bicubic}, types::*};
use nalgebra::{self as na, Point2};

#[derive(Copy, Clone, Debug)]
pub enum MatchingCost {
    SSD, // Sum of Squared Distances
    LSSD // Locally-scaled Sum of Squared Distances     
}

pub trait Patch<const SIZE: usize> {

    const PATTERN_RAW: [[f32; 2]; SIZE];

    /// Returns patch indexes as an array
    fn raw_indexes<'a>() -> &'a [[f32; 2]; SIZE]; 

    fn center(&self) -> &na::Point2<Float>;

    fn datas(&self) -> &[f32; SIZE];
    
    fn jacobian(&self) -> &na::SMatrix<Float, SIZE, 3>;
    
    fn inverse_hessian(&self) -> &na::Matrix3<Float>;

    fn shifted_indexes(&self, transform: &na::Isometry2<Float>) -> [[f32; 2]; SIZE]
    {
        let center = self.center();
        let mut indexes = [[0.0f32; 2]; SIZE];
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
        let center = self.center();
        let center_point = Point2::new(
            center[0], 
            center[1]
        );
        let pixel_coords = transform * center_point;
        let mut index = [0.0f32; 2];
        index[0] = pixel_coords[0];
        index[1] = pixel_coords[1];
        index
    }

    fn residuals(&self, img: &FloatGrayImage, transform: &na::Isometry2<Float>, matching_cost: MatchingCost) -> nalgebra::SVector<Float, 52> {
        use MatchingCost::*;
        match matching_cost {
            SSD => self.residuals_ssd(img, transform),
            LSSD => self.residuals_lssd(img, transform)
        }
    }

    fn residuals_ssd(&self, img: &FloatGrayImage, transform: &na::Isometry2<Float>) -> nalgebra::SVector<Float, 52> {
        let center = self.center();
        let mut residuals = na::SVector::<_, 52>::zeros();
        for ((shift, res), data_host) in Self::raw_indexes().iter()
            .zip(residuals.iter_mut())
            .zip(self.datas().iter())
        {
            let pixel_coords = Point2::new(
                center[0]+shift[0], 
                center[1]+shift[1]
            );

            let pixel_coords = transform * pixel_coords;
            let data_target = interpolate_bicubic(&pixel_coords, img).unwrap_or(0.0); 
            *res = data_target - data_host;
        }
        residuals
    }

    fn residuals_lssd(&self, img: &FloatGrayImage, transform: &na::Isometry2<Float>) -> nalgebra::SVector<Float, 52> {
        let center = self.center();
        let mut residuals = na::SVector::<_, 52>::zeros();
        for (shift, data_target) in Self::raw_indexes().iter()
            .zip(residuals.iter_mut())
        {
            let pixel_coords = Point2::new(
                center[0]+shift[0], 
                center[1]+shift[1]
            );

            let pixel_coords = transform * pixel_coords;
            *data_target = interpolate_bicubic(&pixel_coords, img).unwrap_or(0.0);
        }
        residuals /= residuals.mean();

        for (res, data_host) in residuals.iter_mut().zip(self.datas().iter())
        {
            *res -= data_host;
        }
        residuals
    }

    fn compute_intensities_and_jacobian(
        img: &FloatGrayImage, center: &na::Point2<Float>, matching_cost: MatchingCost
    ) -> ([Float; 52], na::SMatrix<Float, 52, 3>)
    {
        use MatchingCost::*;
        match matching_cost {
            SSD => Self::compute_intensities_and_jacobian_ssd(img, center),
            LSSD => Self::compute_intensities_and_jacobian_lssd(img, center)
        }
    }


    fn compute_intensities_and_jacobian_ssd(
        img: &FloatGrayImage, center: &na::Point2<Float>
    ) -> ([Float; 52], na::SMatrix<Float, 52, 3>)
    {

        let mut intensities = [0.0; 52];
        let mut dr_dtwist = na::SMatrix::<_, 52, _>::zeros(); 

        let mut dpixel_dtwist = na::Matrix2x3::new(
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        );
        
        let mut dimg_dpixel = RowVec2::zeros();
        for ((shift, mut dri_dtwist), intensity) in Self::raw_indexes().iter()
            .zip(dr_dtwist.row_iter_mut())
            .zip(intensities.iter_mut())
        {
            // ri(twist) = img(pixel_i(twist)) - img_target
            // pixel_i(twist) = exp(twist)*pi
            
            // dri_dtwist(0) = dimg_dpixel(pi) * dtransform_dtwist
            // dpixel_dtwist = [
            //   [ -p2  1  0 ],
            //   [  p1  0  1 ]
            //  ]
            
            let pixel_coords = [
                center[0]+shift[0], 
                center[1]+shift[1]
            ];

            dpixel_dtwist[(0,0)] = -pixel_coords[1];
            dpixel_dtwist[(1,0)] =  pixel_coords[0];

            *intensity = d_interpolate_bicubic(&pixel_coords, img, &mut dimg_dpixel)
                .unwrap_or(0.0);

            // dri_dtwist = dimg_dpixel * dtransform_dtwist;
            dimg_dpixel.mul_to(&dpixel_dtwist, &mut dri_dtwist);

            
        }
        
        (intensities, dr_dtwist)
    }


    fn compute_intensities_and_jacobian_lssd(
        img: &FloatGrayImage, center: &na::Point2<Float>
    ) -> ([Float; 52], na::SMatrix<Float, 52, 3>)
    {

        let mut intensities = [0.0; 52];
        let mut mean_intensity = 0.0;
        let mut dintensities_dpixel = na::SMatrix::<Float, 52, 2>::zeros();
        let mut mean_dintensity_dpixel = RowVec2::zeros();
        
        
        let mut dimg_dpixel = RowVec2::zeros();
        for ((shift, mut dintensity_dpixel), intensity) in Self::raw_indexes().iter()
            .zip(dintensities_dpixel.row_iter_mut())
            .zip(intensities.iter_mut())
        {
            let pixel_coords = [center[0]+shift[0], center[1]+shift[1]];
            *intensity = d_interpolate_bicubic(&pixel_coords, img, &mut dimg_dpixel)
                .unwrap_or(0.0);
            dintensity_dpixel.set_row(0, &dimg_dpixel);
            mean_intensity += *intensity;
            mean_dintensity_dpixel += dimg_dpixel;
        }

        mean_intensity /= SIZE as Float;
        mean_dintensity_dpixel /= SIZE as Float;

        let mut dr_dtwist = na::SMatrix::<Float, 52, 3>::zeros(); 
        let mut dpixel_dtwist = na::Matrix2x3::new(
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        );

        for (((shift, intensity), dintensity_dpixel), mut dri_dtwist) in Self::raw_indexes().iter()
            .zip(intensities.iter_mut())
            .zip(dintensities_dpixel.row_iter())
            .zip(dr_dtwist.row_iter_mut())
        {
            let pixel_coords = [center[0]+shift[0], center[1]+shift[1]];
            dpixel_dtwist[(0,0)] = -pixel_coords[1];
            dpixel_dtwist[(1,0)] =  pixel_coords[0];


            let dr_dpixel = (dintensity_dpixel*mean_intensity - (*intensity)*mean_dintensity_dpixel) / mean_intensity.powi(2);

            dr_dpixel.mul_to(&dpixel_dtwist, &mut dri_dtwist);
        }
        
        (intensities, dr_dtwist)
    }
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

pub struct Patch52 {
    center_coords: na::Point2<Float>,
    intensities: [f32; 52],
    dr_dtwist: na::SMatrix<Float, 52, 3>,
    inverse_hessian: na::Matrix3<Float>
}

use std::ops::Not;
impl Patch52 {
    pub fn new(img: &FloatGrayImage, center: &na::Point2<Float>, lm_lambda: Float, matching_cost: MatchingCost) -> Patch52
    {
        let (intensities, dr_dtwist) = Self::compute_intensities_and_jacobian(img, center, matching_cost);

        // hessian = lm_lambda*I + dr_dtwist.transpose() * dr_dtwist
        let mut hessian = na::Matrix3::from_diagonal_element(lm_lambda);
        hessian.gemm(1.0, &dr_dtwist.transpose(), &dr_dtwist, 1.0);

        if hessian.try_inverse_mut().not() {
            panic!("Hessian should be invertible by its definition")
        };
        let inverse_hessian = hessian;

        Self { center_coords: *center, intensities, dr_dtwist, inverse_hessian }
    }
}

impl Patch<52> for Patch52 {
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
    fn raw_indexes<'a>() -> &'a [[f32; 2]; 52] {
        &Self::PATTERN_RAW //.map(|coords| coords.map(|c| c / 2.0))
    }

    fn center(&self) -> &nalgebra::Point2<Float> {
        &self.center_coords
    }

    fn datas(&self) -> &[f32; 52] {
        &self.intensities
    }

    fn jacobian(&self) -> &nalgebra::SMatrix<Float, 52, 3> {
        &self.dr_dtwist
    }

    fn inverse_hessian(&self) -> &nalgebra::Matrix3<Float> {
        &self.inverse_hessian
    }
}



impl DensePatch<52, 8> for Patch52 {
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