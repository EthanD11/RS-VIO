pub mod feature_tracker;
pub mod players;
pub mod ext;
pub mod viewer;
pub mod patch;
pub mod image_operations;

pub mod types {

    use image::{ImageBuffer, Luma};
    use nalgebra as na;
    
    pub type Float = f32;
    pub type FloatGreyImage = ImageBuffer<Luma<Float>, Vec<Float>>; 
    pub type Pyramid = Vec<FloatGreyImage>;

    pub type Vec2 = na::Vector2::<Float>;

    #[cfg(test)]
    mod tests {
        use super::*;
        // use nalgebra as na;

        #[test]
        fn test_vec2() {
            let v = na::Vector2::<f32>::x_axis();
            let v = Vec2::x_axis();
            let vslice = v.as_slice();
        }
    }
}
