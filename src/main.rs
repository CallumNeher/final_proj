use ndarray::prelude::*;
use std::error::Error;
use csv::Reader;
use std::collections::HashMap;
use std::collections::VecDeque;
use rand::Rng;
use std::ops::AddAssign;
use std::env;
use std::io;
use std::io::Write;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

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
            x.push(context.clone());
            y.push(ind.clone());
            context.pop_back();
            context.push_front(ind.clone());
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
    let mut loss = 0.0;
    for (index, &corr_output) in y_batch.iter().enumerate() {
        loss -= (probs[(index, corr_output)]+ 1e-10).ln();
    }
    loss / y_batch.len() as f64
    
}
//trying a new initialization
fn he_initialization(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let limit = 5.0 / (3.0 * (rows as f64).sqrt());
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-limit..limit))
}

fn tanh<D>(input: Array<f64, D>) -> Array<f64, D> where D: ndarray::Dimension{
    input.mapv(|x| x.tanh())
}
fn subtract_row_max(input: &mut Array2<f64>) {
    let max_vals = input.map_axis(Axis(1), |row| {
        row.fold(f64::NEG_INFINITY, |max, &val| f64::max(max, val))
    });
    for (mut row, &max_val) in input.rows_mut().into_iter().zip(max_vals.iter()){
        row.mapv_inplace(|x| x-max_val);
    }
}


#[derive(Clone, Debug)]
struct Network {
    lookup_table: Array2<f64>, //C
    input_weights: Array2<f64>, //W1
    output_weights: Array2<f64>, //W2
    output_bias: Array1<f64>, //b2
}

impl Network {
    fn new() -> Network {
        let mut rng = rand::thread_rng();
        let lookup_table = he_initialization(27,DIMENSIONS);
        let input_weights = he_initialization(BLOCK_SIZE*DIMENSIONS, HIDDEN_SIZE);
        let output_weights = he_initialization(HIDDEN_SIZE,27);
        let output_bias = Array1::from_elem((27), 0.0);
        Network {lookup_table,input_weights,output_weights, output_bias}
    }

    fn embed(&self, labels: VecDeque<usize>) -> Vec<Vec<f64>> {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for &elem in labels.iter() {
            let embedding = self.lookup_table.slice(s![elem, .. ]).to_owned();
            rows.push(embedding.to_vec());
        }
    rows
    }
    fn unembed(&self, embedded_arr: Array2<f64>) -> Vec<Vec<Vec<f64>>> {
        let mut unembedded: Vec<Vec<Vec<f64>>> = Vec::new();
        for row in embedded_arr.axis_iter(Axis(0)){
            let mut big_vec = Vec::new();
            let mut row_as_vec = row.to_vec();
            for i in 0..BLOCK_SIZE{
                let new_row = &row_as_vec[DIMENSIONS*i..DIMENSIONS*(i+1)];
                big_vec.push(new_row.to_vec());
            }
            unembedded.push(big_vec);
        }
    unembedded
    }

    fn forward_pass(&self, batch: Vec<VecDeque<usize>>) -> (Array2<f64>,Array2<f64>,Array2<f64>) {
        let mut emb_vec: Vec<Vec<f64>> = Vec::new();
        for x in batch.iter(){
            let embedding = self.embed(x.clone());
            emb_vec.push(embedding.into_iter().flatten().collect());
        }
        let emb_array: Array2<f64> = Array2::from_shape_vec((BATCH_SIZE, BLOCK_SIZE*DIMENSIONS ), emb_vec.into_iter().flatten().collect()).unwrap();
        let mut hpreact: Array2<f64> = emb_array.clone().dot(&self.input_weights);
        let h = tanh(hpreact);
        let mut logits = h.dot(&self.output_weights) + &self.output_bias;
        subtract_row_max(&mut logits);
        let counts = logits.clone().mapv(|x| x.exp());
        let mut probs = counts.clone();
        for mut row in probs.axis_iter_mut(Axis(0)) {
            let row_sum: f64 = row.iter().sum::<f64>().clone();
            row /= row_sum;
        }
        return (emb_array,h,probs)
    }
    fn backwards_pass(&self, probs: Array2<f64>, h: Array2<f64>, emb_array: Array2<f64>, y_batch: Vec<usize>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>){
        let mut grad_logits = probs.clone();
        for (index, corr_output) in y_batch.iter().enumerate(){
            grad_logits[(index, corr_output.clone())] -= 1.0; // subtracting 1 from targets to backpropagate over cross entropy loss
        }
        //grad_logits.mapv_inplace(|x| x/ BATCH_SIZE as f64);
        let grad_h = grad_logits.dot(&self.output_weights.t());
        let grad_output_weights = h.t().dot(&grad_logits); 
        let grad_output_bias = grad_logits.sum_axis(Axis(0));
        let mut grad_hpreact = &h*&h;
        grad_hpreact.mapv_inplace(|x| 1.0 -x);
        grad_hpreact = &grad_hpreact * &grad_h; // backwards through tanh
        let grad_input_weights = emb_array.t().dot(&grad_hpreact);
        let grad_embedding = grad_hpreact.dot(&self.input_weights.t());
        
        return (grad_embedding,grad_input_weights,grad_output_weights,grad_output_bias)
    }
    fn update_lookup(&mut self, learning_rate:f64, grad_embedding: Array2<f64>, x_batch: Vec<VecDeque<usize>>, mut emb_array:Array2<f64>){
        for (outer_ind,context) in x_batch.iter().enumerate(){
            for (inner_ind,label) in context.iter().enumerate(){
                let grad = grad_embedding.slice(s![outer_ind,inner_ind*DIMENSIONS..(inner_ind+1)*DIMENSIONS]);
                self.lookup_table.scaled_add(-learning_rate, &grad);
            }
        }
    }
//checked
    fn update_rest(&mut self, learning_rate:f64, gradients: (Array2<f64>, Array2<f64>, Array1<f64>)) {
        let (mut grad_input_weights, mut grad_output_weights, mut grad_output_bias) = gradients; 
        self.input_weights.scaled_add(-learning_rate,&grad_input_weights);
        self.output_weights.scaled_add(-learning_rate,&grad_output_weights);
        self.output_bias.scaled_add(-learning_rate,&grad_output_bias)
    }

    fn sample_from_net(&self, input: String, alphabet: Vec<String>){
        let (ind_to_char, char_to_ind) = create_tokenizers(alphabet);
        let mut context: VecDeque<usize> = VecDeque::new();
        let mut out_vec: Vec<usize> = Vec::new();
        for _ in 0..BLOCK_SIZE {
            context.push_front(0 as usize);
        }
        for character in input.chars() {
            let letter = character.to_string();
            let ind = char_to_ind.get(&letter).expect("couldn't find input character in tokenizer");
            context.push_front(ind.clone());
            context.pop_back();
            out_vec.push(*ind);
            //println!("context init: {:?}", &context);
        }
        let mut rng = thread_rng();
        loop { //essentially a repeated forward pass and context update
            let embedding = self.embed(context.clone());
            let emb_array: Array2<f64> = Array2::from_shape_vec((1,BLOCK_SIZE*DIMENSIONS), embedding.into_iter().flatten().collect()).unwrap();
            let mut hpreact: Array2<f64> = emb_array.dot(&self.input_weights);
            let h = tanh(hpreact);
            let mut logits = h.dot(&self.output_weights) + &self.output_bias;
            let counts = logits.clone().mapv(|x| x.exp());
            let mut probs = counts.clone();
            for mut row in probs.axis_iter_mut(Axis(0)) {
                let row_sum: f64 = row.iter().sum::<f64>().clone();
                row /= row_sum;
            }
            let probs_vec = probs.into_raw_vec_and_offset();
            let dist = WeightedIndex::new(&probs_vec.0).expect("Unable to create probability sampling dist");
            let index = dist.sample(&mut rng);
            out_vec.push(index.clone());
            context.push_front(index.clone());
            context.pop_back();
            //println!("context update: {:?}", &context);
            if index == 0{
                break
            }
        }
        //println!("out vec: {:?}", out_vec);
        let mut return_string = String::default();
        for val in out_vec.iter(){
            let letter = ind_to_char.get(&val).unwrap();
            return_string = return_string + &letter;
        }
        println!("New name: {:?}", &return_string);
    }
}

fn read_input(prompt: &str) -> String { //from a prev hw assignmnet of mine
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input");
    input.trim().to_lowercase()
}



fn main() {
    let alphab = load_data("Alphabet Set.csv").expect("Unable to load alphabet");
    let mut list = load_data("Big Names.csv").expect("Unable to load Names");
    let mut nnet = Network::new();
    let (x,y) = build_dataset(alphab.clone(), list);
    let mut learning_rate = 0.01;
    //println!("{:?}", &nnet);    

    for epoch in 0..BATCHES{
        let (x_batch,y_batch) = construct_batch(&x,&y);
        let (emb_array,h,probs) = nnet.forward_pass(x_batch.clone());
        let bp = nnet.backwards_pass(probs.clone(),h,emb_array.clone(),y_batch.clone());
        let loss = cross_entropy_loss(probs,y_batch);
        nnet.update_lookup(learning_rate.clone(), bp.0, x_batch, emb_array);
        nnet.update_rest(learning_rate.clone(), (bp.1,bp.2,bp.3));
        if epoch == BATCHES/2 {
            learning_rate = learning_rate*0.1;
        }
        if epoch%10000 == 0{
            println!("Epoch: {:?}, Loss: {:?}", epoch, loss)
        }

    }
    //println!("{:?}", &nnet);
    loop{
        let input: String = read_input("Input Starting Letters (type 'end' to exit): ");
        if input == "end" {
            break
        }
        nnet.sample_from_net(input, alphab.clone());
    }

}

const BLOCK_SIZE: usize = 3;
const BATCH_SIZE: usize = 32;
const DIMENSIONS: usize = 10;
const HIDDEN_SIZE: usize = 200;
const BATCHES: usize = 200000;

// test ideas: ensure indexing into the lookup table is working correctly (embeddig func)