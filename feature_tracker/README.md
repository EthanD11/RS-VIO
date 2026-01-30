# Minimal Feature Tracker in Rust

## Goal 
Implement a minimal feature tracker in Rust using feature detection (Harris/Fast) and Lukas-Kanade style tracking with $SE(2)$ motion.

## External crates usage
1. [image](https://docs.rs/image/latest/image/) for basic image operations:
    - Module `imageops`:
        - `resize` for image resizing - mainly computing image pyramid (Optionnally using `fast_blur` + raw downsampling for computing image pyramid)
        - `fast_blur` for performing fast blur on images 
2. [imageproc](https://docs.rs/imageproc/latest/imageproc) for more advanced image operations. There is no parallel versions of separable filters (multi-threading) and no gpu support:
    - Module `corners`: For the `FAST` corner detector
    - Module `suppress`: For performing non-maximum suppression (for the Harris corner detector)
    - Module `integral_image`: For fast computation of rectangular averages (for the Harris corner detector)
3. [nalgebra](https://docs.rs/nalgebra/latest/nalgebra/) for small matrix/vector and Lie Groups $SE(2)$ computations

## Plan
1. [ ]  Feature detection