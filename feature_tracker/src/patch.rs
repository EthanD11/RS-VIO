
use crate::{image_operations::interpolate_bicubic, types::*};
use nalgebra::{self as na, ArrayStorage, U3, U52};

pub trait Patch<const SIZE: usize> {

    const PATTERN_RAW: [[f32; 2]; SIZE];

    /// Returns patch indexes as an array
    fn indexes() -> [[f32; 2]; SIZE]; 

    fn intensities(&self) -> [f32; SIZE];
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
    coords: [f32; 2],
    intensities: [f32; 52]
}

impl<'a> Patch52<'a> {
    pub fn new(img: &'a FloatGrayImage, coords: &[f32; 2]) -> Patch52<'a>
    {
        let mut intensities = [0.0; 52];
        for (point, intensity) in Self::PATTERN_RAW.iter().zip(intensities.iter_mut()) {
            *intensity = interpolate_bicubic(point, img).unwrap_or(0.0);
        }
        Self { img, coords: *coords, intensities }
    }

    fn compute_se2_jacobian(&self) -> na::Matrix<Float, U52, U3, ArrayStorage<Float, 52, 3>>{
        todo!()
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
    fn indexes() -> [[f32; 2]; 52] {
        Self::PATTERN_RAW
    }

    fn intensities(&self) -> [f32; 52] {
        self.intensities
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