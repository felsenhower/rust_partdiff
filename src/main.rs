use std::env;
use std::ops::{Index, IndexMut};
use std::process;
use std::time::{Duration, Instant};
use std::vec;

// The supported calculation Algorithms
// Gauss Seidel working on the same matrix
// Jacobi using in and out matrices
#[derive(Debug, PartialEq)]
enum CalculationMethod {
    MethGaussSeidel,
    MethJacobi,
}

// For parsing command line arguments
impl std::str::FromStr for CalculationMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "MethGaussSeidel" | "1" => Ok(CalculationMethod::MethGaussSeidel),
            "MethJacobi" | "2" => Ok(CalculationMethod::MethJacobi),
            _ => Err(format!(
                "'{}' is not a valid value for CalculationMethod",
                s
            )),
        }
    }
}

// The supported inference functions used during calculation
// F0:     f(x,y) = 0
// FPiSin: f(x,y) = 2pi^2*sin(pi*x)sin(pi*y)
#[derive(Debug, PartialEq)]
enum InferenceFunction {
    FuncF0,
    FuncFPiSin,
}

// For parsing command line arguments
impl std::str::FromStr for InferenceFunction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "FuncF0" | "1" => Ok(InferenceFunction::FuncF0),
            "FuncFPiSin" | "2" => Ok(InferenceFunction::FuncFPiSin),
            _ => Err(format!(
                "'{}' is not a valid value for InferenceFunction",
                s
            )),
        }
    }
}

// The supported termination conditions
// TermAcc: terminate after set accuracy is reached
// TermIter: terminate after set amount of iterations
#[derive(Debug, PartialEq)]
enum TerminationCondition {
    TermAcc,
    TermIter,
}

// For parsing command line arguments
impl std::str::FromStr for TerminationCondition {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "TermAcc" | "1" => Ok(TerminationCondition::TermAcc),
            "TermIter" | "2" => Ok(TerminationCondition::TermIter),
            _ => Err(format!(
                "'{}' is not a valid value for TerminationCondition",
                s
            )),
        }
    }
}

// Data structure for storing the given parameters for the calculation
#[derive(Debug)]
struct CalculationOptions {
    _number: u64,                      // number of threads
    method: CalculationMethod,         // Gauss Seidel or Jacobi method of iteration
    interlines: usize,                 // matrix size = interline*8+9
    inf_func: InferenceFunction,       // inference function
    termination: TerminationCondition, // termination condition
    term_iteration: u64,               // terminate if iteration number reached
    term_accuracy: f64,                // terminate if accuracy reached
}

impl CalculationOptions {
    fn new(
        _number: u64,
        method: CalculationMethod,
        interlines: usize,
        inf_func: InferenceFunction,
        termination: TerminationCondition,
        term_iteration: u64,
        term_accuracy: f64,
    ) -> CalculationOptions {
        CalculationOptions {
            _number,
            method,
            interlines,
            inf_func,
            termination,
            term_iteration,
            term_accuracy,
        }
    }
}

// Data structure for storing the the data needed during calculation
#[derive(Debug)]
struct CalculationArguments {
    n: usize,                 // Number of spaces between lines (lines=n+1)
    num_matrices: usize,      // number of matrices
    h: f64,                   // length of a space between two lines
    matrices: PartdiffTensor, // The matrices for calculation
}

impl CalculationArguments {
    fn new(n: usize, num_matrices: usize, h: f64) -> CalculationArguments {
        let matrices: PartdiffTensor = PartdiffTensor::new(n + 1, num_matrices);

        CalculationArguments {
            n,
            num_matrices,
            h,
            matrices,
        }
    }
}

// Data structure for storing result data of the calculation
#[derive(Debug)]
struct CalculationResults {
    m: usize,            // Index of matrix that holds the final state
    stat_iteration: u64, // number of current iteration
    stat_accuracy: f64,  // actual accuracy of all slaces in iteration
}

impl CalculationResults {
    fn new(m: usize, stat_iteration: u64, stat_accuracy: f64) -> CalculationResults {
        CalculationResults {
            m,
            stat_iteration,
            stat_accuracy,
        }
    }
}

// Simple data structure for a 3D matrix [m][i][j]
// Has an efficient continuous 1D memory layout
#[derive(Debug)]
struct PartdiffTensor {
    n: usize,
    matrix: Vec<f64>,
}

impl PartdiffTensor {
    fn new(n: usize, num_matrices: usize) -> PartdiffTensor {
        let matrix = vec![0.0; num_matrices * n * n];
        PartdiffTensor { n, matrix }
    }
}

// Implementation of Index and IndexMut traits for the 3D matrix
// 3d-array-indexing allows access to matrix elements with following syntax:
//   matrix[[m,i,j]]
// Only the unsafe, unchecked variant is kept for maximum performance.
impl Index<[usize; 3]> for PartdiffTensor {
    type Output = f64;

    fn index(&self, idx: [usize; 3]) -> &Self::Output {
        let offset = idx[0] * self.n * self.n + idx[1] * self.n + idx[2];
        unsafe { self.matrix.get_unchecked(offset) }
    }
}

impl IndexMut<[usize; 3]> for PartdiffTensor {
    fn index_mut(&mut self, idx: [usize; 3]) -> &mut Self::Output {
        let offset = idx[0] * self.n * self.n + idx[1] * self.n + idx[2];
        unsafe { self.matrix.get_unchecked_mut(offset) }
    }
}

// Display help message to show the required command line arguments to run the binary
fn usage() {
    println!(
        "Usage: ./rust_partdiff [number] [method] [interlines] [func] [termination] [acc/iter]\n"
    );
    println!("  -number:      number of threads (1 .. n)");
    println!("  -method:      calculation method (MethGaussSeidel/MethJacobi OR 1/2)");
    println!("  -interlines:  number of interlines (1 .. n)");
    println!("                  matrixsize = (interlines * 8) + 9");
    println!("  -func:        inference function (FuncF0/FuncFPiSin OR 1/2)");
    println!("  -termination: termination condition (TermAcc/TermIter OR 1/2)");
    println!("                  TermAcc: sufficient accuracy");
    println!("                  TermIter: number of iterations");
    println!("  -acc/iter:    depending on termination:");
    println!("                  accuracy: 1e-4 .. 1e-20");
    println!("                  iterations: 1 .. n");
}

// Helper function to parse command line arguments
fn parse_arg<U>(arg: Option<String>) -> U
where
    U: std::str::FromStr,
    <U as std::str::FromStr>::Err: std::fmt::Display,
{
    let ret: U = match arg {
        Some(a) => a.parse().unwrap_or_else(|error| {
            eprintln!("Error: {}", error);
            usage();
            process::exit(1);
        }),
        None => {
            eprintln!("Error: incomplete arguments.");
            usage();
            process::exit(1);
        }
    };
    ret
}

// Parsing of command line arguments
fn ask_params(mut args: std::env::Args) -> CalculationOptions {
    // TODO keep authors of original c version?
    // println!("============================================================");
    // println!("Program for calculation of partial differential equations.  ");
    // println!("============================================================");
    // println!("(c) Dr. Thomas Ludwig, TU München.");
    // println!("    Thomas A. Zochler, TU München.");
    // println!("    Andreas C. Schmidt, TU München.");
    // println!("============================================================");

    // TODO interactive arguments

    args.next();

    let number: u64 = parse_arg(args.next());
    if number < 1 {
        eprintln!("Error number argument must be a positive integer");
        usage();
        process::exit(1);
    }

    let method: CalculationMethod = parse_arg(args.next());

    let interlines: usize = parse_arg(args.next());

    let inf_func: InferenceFunction = parse_arg(args.next());

    let termination: TerminationCondition = parse_arg(args.next());

    // Check for the meaning of the last argument
    match termination {
        TerminationCondition::TermAcc => {
            let acc: f64 = parse_arg(args.next());
            if (acc < 1e-20) | (acc > 1e-4) {
                eprintln!("Error: termination accuracy must be between 1e-20 and 1e-4");
                usage();
                process::exit(1);
            }
            return CalculationOptions::new(
                number,
                method,
                interlines,
                inf_func,
                termination,
                std::u64::MAX,
                acc,
            );
        }
        TerminationCondition::TermIter => {
            let iterations = parse_arg(args.next());
            if iterations < 1 {
                eprintln!("Error: termination iterations must be > 0");
                usage();
                process::exit(1);
            }
            return CalculationOptions::new(
                number,
                method,
                interlines,
                inf_func,
                termination,
                iterations,
                0.0,
            );
        }
    }
}

// Determine calculation arguments and initialize calculation results
fn init_variables(options: &CalculationOptions) -> (CalculationArguments, CalculationResults) {
    let n: usize = (options.interlines * 8) + 9 - 1;
    let num_matrices: usize = match options.method {
        CalculationMethod::MethGaussSeidel => 1,
        CalculationMethod::MethJacobi => 2,
    };
    let h: f64 = 1.0 as f64 / n as f64;
    let arguments = CalculationArguments::new(n, num_matrices, h);
    let results = CalculationResults::new(0, 0, 0.0);

    (arguments, results)
}

// Initialize the matrix borders according to the used inference function
fn init_matrices(arguments: &mut CalculationArguments, options: &CalculationOptions) {
    if options.inf_func == InferenceFunction::FuncF0 {
        let matrix = &mut arguments.matrices;
        let n = arguments.n;
        let h = arguments.h;

        for g in 0..arguments.num_matrices as usize {
            for i in 0..(n + 1) {
                matrix[[g, i, 0]] = 1.0 - (h * i as f64);
                matrix[[g, i, n]] = h * i as f64;
                matrix[[g, 0, i]] = 1.0 - (h * i as f64);
                matrix[[g, n, i]] = h * i as f64;
            }
        }
    }
}

// Main calculation
fn calculate(
    arguments: &mut CalculationArguments,
    results: &mut CalculationResults,
    options: &CalculationOptions,
) {
    const PI: f64 = 3.141592653589793;
    const TWO_PI_SQUARE: f64 = 2.0 * PI * PI;

    let n = arguments.n;
    let h = arguments.h;

    let mut star: f64;
    let mut residuum: f64;
    let mut maxresiduum: f64;

    let mut pih: f64 = 0.0;
    let mut fpisin: f64 = 0.0;

    let mut term_iteration = options.term_iteration;

    // for distinguishing between old and new state of the matrix if two matrices are used
    let mut m1: usize = 0;
    let mut m2: usize = 0;

    if options.method == CalculationMethod::MethJacobi {
        m1 = 0;
        m2 = 1;
    }

    if options.inf_func == InferenceFunction::FuncFPiSin {
        pih = PI * h;
        fpisin = 0.25 * TWO_PI_SQUARE * h * h;
    }

    while term_iteration > 0 {
        let matrix = &mut arguments.matrices;

        maxresiduum = 0.0;

        for i in 1..n {
            let mut fpisin_i = 0.0;

            if options.inf_func == InferenceFunction::FuncFPiSin {
                fpisin_i = fpisin * (pih * i as f64).sin();
            }

            for j in 1..n {
                star = 0.25
                    * (matrix[[m2, i - 1, j]]
                        + matrix[[m2, i, j - 1]]
                        + matrix[[m2, i, j + 1]]
                        + matrix[[m2, i + 1, j]]);

                if options.inf_func == InferenceFunction::FuncFPiSin {
                    star += fpisin_i * (pih * j as f64).sin();
                }

                if (options.termination == TerminationCondition::TermAcc) | (term_iteration == 1) {
                    residuum = (matrix[[m2, i, j]] - star).abs();

                    maxresiduum = match residuum {
                        r if r < maxresiduum => maxresiduum,
                        _ => residuum,
                    };
                }

                matrix[[m1, i, j]] = star;
            }
        }

        results.stat_iteration += 1;
        results.stat_accuracy = maxresiduum;

        let tmp = m1;
        m1 = m2;
        m2 = tmp;

        match options.termination {
            TerminationCondition::TermAcc => {
                if maxresiduum < options.term_accuracy {
                    term_iteration = 0;
                }
            }
            TerminationCondition::TermIter => term_iteration -= 1,
        }
    }

    results.m = m2;
}

fn format_residuum(x: f64) -> String {
    let s = format!("{:.6e}", x);
    let (base, exp_str) = s.split_once('e').unwrap();
    let exp: i32 = exp_str.parse().unwrap();
    format!("{base}e{:+03}", exp)
}

// Display important information about the calculation
fn display_statistics(
    arguments: &CalculationArguments,
    results: &CalculationResults,
    options: &CalculationOptions,
    duration: Duration,
) {
    let n = arguments.n;

    println!("Calculation time:       {:.6} s", duration.as_secs_f64());
    println!(
        "Memory usage:           {:.6} MiB",
        ((n + 1) * (n + 1) * std::mem::size_of::<f64>() * arguments.num_matrices) as f64
            / 1024.0
            / 1024.0
    );
    println!(
        "Calculation method:     {}",
        match options.method {
            CalculationMethod::MethGaussSeidel => "Gauß-Seidel",
            CalculationMethod::MethJacobi => "Jacobi",
        }
    );
    println!("Interlines:             {}", options.interlines);
    println!(
        "Perturbation function:  {}",
        match options.inf_func {
            InferenceFunction::FuncF0 => "f(x,y) = 0",
            InferenceFunction::FuncFPiSin => "f(x,y) = 2 * pi^2 * sin(pi * x) * sin(pi * y)",
        }
    );
    println!(
        "Termination:            {}",
        match options.termination {
            TerminationCondition::TermAcc => "Required accuracy",
            TerminationCondition::TermIter => "Number of iterations",
        }
    );
    println!("Number of iterations:   {}", results.stat_iteration);
    println!(
        "Residuum:               {}",
        format_residuum(results.stat_accuracy)
    );
}

// Beschreibung der Funktion displayMatrix:
//
// Die Funktion displayMatrix gibt eine Matrix
// in einer "ubersichtlichen Art und Weise auf die Standardausgabe aus.
//
// Die "Ubersichtlichkeit wird erreicht, indem nur ein Teil der Matrix
// ausgegeben wird. Aus der Matrix werden die Randzeilen/-spalten sowie
// sieben Zwischenzeilen ausgegeben.
fn display_matrix(
    arguments: &mut CalculationArguments,
    results: &CalculationResults,
    options: &CalculationOptions,
) {
    let matrix = &arguments.matrices;
    let interlines = options.interlines;

    println!("");
    println!("Matrix:");
    for y in 0..9 as usize {
        for x in 0..9 as usize {
            print!(
                " {:.4}",
                matrix[[results.m, y * (interlines + 1), x * (interlines + 1)]]
            );
        }
        print!("\n");
    }
}

fn main() {
    let options = ask_params(env::args());
    let (mut arguments, mut results) = init_variables(&options);
    init_matrices(&mut arguments, &options);
    let now = Instant::now();
    calculate(&mut arguments, &mut results, &options);
    let duration = now.elapsed();
    display_statistics(&arguments, &results, &options, duration);
    display_matrix(&mut arguments, &results, &options);
}
