
use crate::types::*;

trait Patch<const SIZE: usize> {
    
    const PATTERN_RAW: [[f32; 2]; SIZE];

    /// Returns patch indexes as an array
    fn indexes() -> [[f32; 2]; SIZE]; 

    fn intensities(&self, pyramid_level: usize) -> [f32; SIZE];
}

trait DensePatch<const SIZE: usize, const NROWS: usize>: Patch<SIZE> {
    /// Returns patch indexes as rows
    /// This assumes the patch can be described as follows:
    /// 
    /// Row i: `[[x_start_i, y_start_i], [x_end_i, y_end_i]]`
    /// 
    /// where all the pixels between start_i and end_i are part of the patch.
    fn rows() ->  [[[f32; 2]; 2]; NROWS];
}

pub struct Patch52<'a> {

    pyramid: &'a Pyramid

}

impl<'a> Patch52<'a> {
    

    pub fn new(pyramid: &'a Pyramid) -> Patch52<'a>
    {
        Self { pyramid }
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

    fn intensities(&self, pyramid_level: usize) -> [f32; 52] {
        let image_level = self.pyramid.get(pyramid_level).unwrap();
        todo!()
        // image_level
    }
}
