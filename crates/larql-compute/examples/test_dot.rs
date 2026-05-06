use ndarray::Array2;

fn main() {
    let m = 256;
    let k = 1024;
    let a = Array2::<f32>::zeros((m, k));
    let x = Array2::<f32>::zeros((1, k));
    println!("Running dot product...");
    let y = x.dot(&a.t());
    println!("Result shape: {:?}", y.shape());
}
