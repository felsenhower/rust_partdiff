# rust_partdiff
Program for calculation of partial differential equations implemented in Rust

# building the crate
To build the crate correctly currently some features have to be activated.
## Features
* Decide which indexing implementation is used for PartdiffMatrix
  * feature: `C-style-indexing` has a syntax of `matrix[x][y]` and has probably worse performance due to the internals of the implementation 
  * not building with `C-style-indexing` uses `2d-array-indexing` which has a syntax of `matrix[[x,y]]` and should provide better performance
* Decide whether bounds checking should be used for matrix access
  * feature: `unsafe-indexing` uses the unsafe `get_unchecked` methods of Vec and does no bounds checking but therefore performs better
  * not building with `unsafe-indexing` defaults to using access methods of Vec that apply bounds checking

The current default way to build the crate for performance should be: `cargo build --release --features "unsafe-indexing"`
