use std::{fs::File, io::Write};

use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::Array1;

fn main() {
    // Load the Iris dataset
    let (train, test) = linfa_datasets::iris().split_with_ratio(0.8);

    // Initialize the Decision Tree model
    let model = DecisionTree::params()
        .max_depth(Some(3))
        .split_quality(linfa_trees::SplitQuality::Gini);

    // Fit the model to the training data
    let decision_tree = model.fit(&train).unwrap();

    // Predict the labels of the test data
    let predictions: Array1<usize> = decision_tree.predict(test.records());

    // Evaluate the model
    let accuracy = predictions.confusion_matrix(&test).unwrap().accuracy();

    println!("Predicitions: {:?}", predictions);
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    // Export trained tree graph to Tikz
    let tikz = decision_tree.export_to_tikz();
    let latex_tree: String = tikz.to_string();

    // Save trained tree graph in LaTex
    let mut f = File::create("Decisio Tree Graph.tex").unwrap();
    f.write_all(latex_tree.as_bytes()).unwrap();
}