use ndarray::prelude::*;
use std::error::Error;
use csv::Reader;
use std::collections::HashMap;
use std::collections::VecDeque;
use rand::Rng;
use std::ops::AddAssign;

// Load Data in

fn load_data(path:&str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut rdr = Reader::from_path(path)?;
    let mut words = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let slice = record.as_slice().to_string();
        let word = slice;
        words.push(word);
    }
    Ok(words)
}

// construct a character tokenizer

fn create_tokenizers(alphabet: Vec<String>)-> (HashMap<usize,String>, HashMap<String,usize>) {
    let mut ind_char = HashMap::new();
    let mut char_ind = HashMap::new();
    char_ind.insert(".".to_string(),0);
    ind_char.insert(0, ".".to_string());
    for (i,a) in alphabet.iter().enumerate() {
        ind_char.insert(i+1, a.clone());
        char_ind.insert(a.clone(),i+1);
    }
    (ind_char, char_ind)
}

//build Dataset

fn build_dataset(alphabet: Vec<String>, mut words: Vec<String>)-> (Vec<VecDeque<usize>>, Vec<usize>) {
    let (ind_to_char, char_to_ind) = create_tokenizers(alphabet);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for word in words.iter_mut() {
        word.make_ascii_lowercase();
        word.push_str(".");
        let mut context: VecDeque<usize> = VecDeque::new();
        for _ in 0..BLOCK_SIZE {
            context.push_front(0 as usize);
        }
        for character in word.chars() {
            let lett = character.to_string();
            let ind = char_to_ind.get(&lett).expect("couldn't find character in tokenizer");
            context.pop_back();
            context.push_front(ind.clone());
            x.push(context.clone());
            y.push(ind.clone());
        }
    }
    (x,y)
}

fn construct_batch(data: &Vec<VecDeque<usize>>, y: &Vec<usize>) -> (Vec<VecDeque<usize>>, Vec<usize>) {
    let mut rng = rand::thread_rng();
    let batch_indices: Vec<usize> = (0..BATCH_SIZE).map(|_| rng.gen_range(0..data.len())).collect();
    let mut x_batch: Vec<VecDeque<usize>> = Vec::new();
    let mut y_batch: Vec<usize> = Vec::new();
    for i in batch_indices.iter(){
        x_batch.push(data.get(i.clone()).unwrap().clone());
        y_batch.push(y.get(i.clone()).unwrap().clone());
    }
    return (x_batch,y_batch)
}

fn cross_entropy_loss(probs: Array2<f64>, y_batch: Vec<usize>) -> f64 {
    let mut probs_correct: Vec<f64> = Vec::new();
    for (index, corr_output) in y_batch.iter().enumerate() {
        probs_correct.push(probs[(index.clone(), corr_output.clone())]);
    }
    let loss = -probs_correct.iter().map(|x| x.ln()).clone().sum::<f64>() / (probs_correct.len().clone() as f64);
    loss
    
}
#[derive(Clone)]
struct Network {
    lookup_table: Array2<f64>, //C
    input_weights: Array2<f64>, //W1
    output_weights: Array2<f64>, //W2
    output_bias: Array1<f64>, //b2
}

impl Network {
    fn new() -> Network {
        let mut rng = rand::thread_rng();
        let lookup_table = Array2::from_shape_fn((27, DIMENSIONS), |_| rng.gen_range(0.0..1.0));
        let input_weights = Array2::from_shape_fn((BLOCK_SIZE*DIMENSIONS, HIDDEN_SIZE), |_| rng.gen_range(0.0..1.0));
        let output_weights = Array2::from_shape_fn((HIDDEN_SIZE, 27), |_| rng.gen_range(0.0..1.0));
        let output_bias = Array1::from_shape_fn(27, |_| rng.gen_range(0.0..1.0));
        Network {lookup_table,input_weights,output_weights, output_bias}
    }
    fn sigmoid(x: Array2<f64>) -> Array2<f64> {
        x.mapv(|z| 1.0 / (1.0 + (-z).exp()))
    }

    fn embed(&self, labels: VecDeque<usize>) -> Vec<Vec<f64>> {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for &elem in labels.iter() {
            let embedding = self.lookup_table.slice(s![elem, .. ]).to_owned();
            rows.push(embedding.to_vec());
        }
    rows
    }

    fn forward_pass(&self, batch: Vec<VecDeque<usize>>) -> (Array2<f64>,Array2<f64>) {
        let mut emb_vec: Vec<Vec<f64>> = Vec::new();
        for x in batch.iter(){
            let embedding = self.embed(x.clone());
            emb_vec.push(embedding.into_iter().flatten().collect());
        }
        let emb_array: Array2<f64> = Array2::from_shape_vec((BATCH_SIZE, BLOCK_SIZE*DIMENSIONS ), emb_vec.into_iter().flatten().collect()).unwrap();
        let z1: Array2<f64> = emb_array.dot(&self.input_weights);
        let a1 = Network::sigmoid(z1);
        let z2 = a1.clone().dot(&self.output_weights) + self.output_bias.clone();
        let a2 = Network::sigmoid(z2);
        let mut probs = a2.map(|x| x.exp());
        for mut row in probs.axis_iter_mut(Axis(0)) {
            let row_sum: f64 = row.iter().sum::<f64>().clone();
            row /= row_sum;
        }
        return (probs,a1)
    }
    fn backwards_pass(&self, probs: Array2<f64>, a1: Array2<f64>, y_batch: Vec<usize>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>){
        let mut grad_z2 = probs.clone();
        for (index, corr_output) in y_batch.iter().enumerate(){
            grad_z2[(index, corr_output.clone())] -= 1.0; // subtracting 1 from targets to backpropagate over cross entropy loss
        }
        let grad_a1 = grad_z2.dot(&self.output_weights.t()) * &a1 * (1.0 - &a1); //calculating gradient over softmax (sigmoid)
        let grad_output_weights = grad_z2.t().dot(&a1); 
        let grad_output_bias = grad_z2.sum_axis(Axis(0));
        let grad_input_weights = a1.t().dot(&grad_a1);
        let grad_lookup_table = grad_a1.dot(&self.input_weights.t());
        return (grad_lookup_table,grad_input_weights,grad_output_weights,grad_output_bias)
    }
    fn update(&mut self, learning_rate: f64, gradients: (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>)) {
        let (grad_lookup_table,grad_input_weights,grad_output_weights,grad_output_bias) = gradients;
        self.lookup_table = self.lookup_table.clone() + grad_lookup_table * learning_rate;
        self.input_weights = self.input_weights.clone() + grad_input_weights * learning_rate;
        //self.output_weights = self.output_weights.clone() + grad_output_weights * learning_rate;
        self.output_bias = self.output_bias.clone() + grad_output_bias * learning_rate;
    }
}

fn main() {
    let alphab = load_data("Alphabet Set.csv").expect("Unable to load alphabet");
    let mut list = load_data("1900 Names.csv").expect("Unable to load Names");
    let mut nnet = Network::new();
    let (x,y) = build_dataset(alphab, list);
    
// build training loop:
    for _ in 0..BATCHES{
        let (x_batch,y_batch) = construct_batch(&x,&y);
        let (p,a1) = nnet.clone().forward_pass(x_batch);
        let bp = nnet.backwards_pass(p.clone(),a1,y_batch.clone());
        let loss = cross_entropy_loss(p,y_batch);
        nnet.update(LEARNING_RATE, bp);
        println!("{:?}", loss);
    }
}

const BLOCK_SIZE: usize = 3;
const BATCH_SIZE: usize = 50;
const DIMENSIONS: usize = 10;
const HIDDEN_SIZE: usize = 15;
const BATCHES: usize = 3;
const LEARNING_RATE: f64 = 0.01;

// test ideas: ensure indexing into the lookup table is working correctly (embeddig func)