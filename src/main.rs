use ndarray::prelude::*;
use std::error::Error;
use csv::Reader;
use std::collections::HashMap;
use std::collections::VecDeque;
use rand::Rng;

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

fn construct_batch(data: Vec<VecDeque<usize>>) -> Vec<VecDeque<usize>> {
    let mut rng = rand::thread_rng();
    let batch_indices: Vec<usize> = (0..BATCH_SIZE).map(|_| rng.gen_range(0..data.len())).collect();
    let mut batch: Vec<VecDeque<usize>> = Vec::new();
    for i in batch_indices.iter(){
        let datum = data.get(i.clone()).unwrap();
        batch.push(datum.clone());
    }
    return batch
}

struct Network {
    lookup_table: Array2<f64>, //C
    input_weights: Array2<f64>, //W1
    output_weights: Array2<f64>, //W2
    output_bias: Array1<f64>, //b2
}

impl Network {
    fn new(hidden_size: usize) -> Network {
        let mut rng = rand::thread_rng();
        let lookup_table = Array2::from_shape_fn((27, DIMENSIONS), |_| rng.gen_range(0.0..1.0));
        let input_weights = Array2::from_shape_fn((BLOCK_SIZE*DIMENSIONS, hidden_size), |_| rng.gen_range(0.0..1.0));
        let output_weights = Array2::from_shape_fn((hidden_size, 10), |_| rng.gen_range(0.0..1.0));
        let output_bias = Array1::from_shape_fn(10, |_| rng.gen_range(0.0..1.0));
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

    fn forward_pass(&self, batch: Vec<VecDeque<usize>>) -> Array2<f64> {
        let mut emb_vec: Vec<Vec<f64>> = Vec::new();
        for x in batch.iter(){
            let embedding = self.embed(x.clone());
            emb_vec.push(embedding.into_iter().flatten().collect());
        }
        let emb_array: Array2<f64> = Array2::from_shape_vec((BATCH_SIZE, BLOCK_SIZE*DIMENSIONS ), emb_vec.into_iter().flatten().collect()).unwrap();
        let z1: Array2<f64> = emb_array.dot(&self.input_weights);
        let a1 = Network::sigmoid(z1);
        let z2 = a1.dot(&self.output_weights) + self.output_bias.clone();
        let a2 = Network::sigmoid(z2);
        println!("{:?}", a2);
        return a2
        

    }
}

fn main() {
    let alphab = load_data("Alphabet Set.csv").expect("Unable to load alphabet");
    let mut list = load_data("1900 Names.csv").expect("Unable to load Names");
    let (x,y) = build_dataset(alphab, list);
    let batch = construct_batch(x);
    let nnet = Network::new(5);
    let a = nnet.forward_pass(batch);
    //println!("{:?}",batch);
}

const BLOCK_SIZE: usize = 2;
const BATCH_SIZE: usize = 3;
const DIMENSIONS: usize = 2;

// test ideas: ensure indexing into the lookup table is working correctly (embeddig func)