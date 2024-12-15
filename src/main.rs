use ndarray::prelude::*;
use csv::Reader;
use std::collections::VecDeque;
use rand::Rng;
use std::io;
use std::io::Write;
use plotters::prelude::*;
pub mod network;
use network::Network;
use network::create_tokenizers;
use std::error::Error;

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



//build Dataset

fn build_dataset(alphabet: Vec<String>, mut words: Vec<String>)-> ((Vec<VecDeque<usize>>, Vec<usize>),(Vec<VecDeque<usize>>, Vec<usize>)) {
    let (_, char_to_ind) = create_tokenizers(alphabet);
    let mut xtr = Vec::new();
    let mut ytr = Vec::new();
    let mut xte = Vec::new();
    let mut yte = Vec::new();
    for (ind,word) in words.iter_mut().enumerate() {
        word.make_ascii_lowercase();
        word.push_str(".");
        if ind %10 == 0{
            let mut context: VecDeque<usize> = VecDeque::new();
            for _ in 0..BLOCK_SIZE {
                context.push_front(0 as usize);
            }
            for character in word.chars() {
                let lett = character.to_string();
                let ind = char_to_ind.get(&lett).expect("couldn't find character in tokenizer");
                xte.push(context.clone());
                yte.push(ind.clone());
                context.pop_back();
                context.push_front(ind.clone());
            }
        }
        else{
            let mut context: VecDeque<usize> = VecDeque::new();
            for _ in 0..BLOCK_SIZE {
                context.push_front(0 as usize);
            }
            for character in word.chars() {
                let lett = character.to_string();
                let ind = char_to_ind.get(&lett).expect("couldn't find character in tokenizer");
                xtr.push(context.clone());
                ytr.push(ind.clone());
                context.pop_back();
                context.push_front(ind.clone());
            }
        }
        
    }
    ((xtr,ytr),(xte,yte))
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

fn read_input(prompt: &str) -> String { //from a prev hw assignmnet of mine
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input");
    input.trim().to_lowercase()
}

fn main() {
    println!("Welcome! Training Network");
    let alphab = load_data("Alphabet Set.csv").expect("Unable to load alphabet");
    let list = load_data("Big Names.csv").expect("Unable to load Names");
    let mut nnet = Network::new();
    let ((xtr,ytr),(xte,yte)) = build_dataset(alphab.clone(), list);
    let mut learning_rate = 0.01; 

    for epoch in 0..BATCHES{
        let (x_batch,y_batch) = construct_batch(&xtr,&ytr);
        let (emb_array,h,probs) = nnet.forward_pass(x_batch.clone());
        let bp = nnet.backwards_pass(probs.clone(),h,emb_array.clone(),y_batch.clone());
        nnet.update_lookup(learning_rate.clone(), bp.0, x_batch);
        nnet.update_rest(learning_rate.clone(), (bp.1,bp.2,bp.3));
        if epoch == BATCHES/2 {
            learning_rate = learning_rate*0.1;
        }
        if epoch%10000 == 0{
            let loss = cross_entropy_loss(probs,y_batch);
            println!("Epoch: {:?}, Loss: {:?}", epoch, loss);
        }
        
    }
    println!("Network Trained");   
    let (_,_,probs) = nnet.forward_pass(xte);
    let f_loss = cross_entropy_loss(probs,yte);
    println!("Final Model Loss: {:?}", f_loss);
    println!("If you wish to generate a name based on starting letters, type them in. Otherwise press enter");
    loop{
        let input: String = read_input("Input Starting Letters (type 'end' to exit): ");
        if input == "end" {
            break
        }
        nnet.sample_from_net(input, alphab.clone());
    }
    println!("Ending Program, Have a good day!");

    let tuple_vec: Vec<(f64,f64)> = nnet.lookup_table.rows().into_iter().map(|row| (row[0],row[1])).collect();
    let placeholder = vec![tuple_vec[0]];
    let vowels = vec![tuple_vec[1],tuple_vec[5],tuple_vec[9],tuple_vec[15],tuple_vec[21]];
    let consonants = tuple_vec;
//PLOT ONLY IMPLEMENTED FOR 2 DIMENSIONAL CHARACTER EMBEDDINGS
// For maximum model accuracy, 10 dimensional embeddings is recommended, but 2 dimensional embeddings are easier to visualize
    let _ = plot(vowels, consonants, placeholder);
}

fn plot(vowels: Vec<(f64,f64)>, consonants: Vec<(f64,f64)>, placeholder: Vec<(f64,f64)>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("visual.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.titled("Plotting Letters on 2 dimensions", ("Arial", 20).into_font())?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Vowels: Green, Consonants: Red", ("Arial", 20).into_font())
        .x_label_area_size(80)
        .y_label_area_size(80)
        .build_cartesian_2d(-1f64..1f64, -1f64..1f64)?;
    chart.configure_mesh().draw()?;
    chart.draw_series(consonants.iter().map(|(x,y)| Circle::new((*x,*y),3,RED.filled())))?;
    chart.draw_series(vowels.iter().map(|(x,y)| Circle::new((*x,*y),3,GREEN.filled())))?;
    chart.draw_series(placeholder.iter().map(|(x,y)| Circle::new((*x,*y),3,BLACK.filled())))?;
    Ok(())
}


const BATCH_SIZE: usize = 32;
const BATCHES: usize = 1;
const BLOCK_SIZE: usize = 3;
