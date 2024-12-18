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
    if DIMENSIONS == 2{
        let _ = plot(vowels, consonants, placeholder);
    }
}

const BATCH_SIZE: usize = 32;
const BATCHES: usize = 200000;
const BLOCK_SIZE: usize = 3;

#[cfg(test)]
mod tests {
    use super::*;
    use network::{subtract_row_max, tanh};
    const DIMENSIONS: usize = 2;
    #[test]
    fn test_forward_pass() { // testing the forward pass logic:
        let test_net = Network::new();
        let t_context = VecDeque::from([0,1,0]);
        let batch = vec![t_context.clone(),t_context];

        //forward pass implementation with tweaked emb_array size:
        let mut emb_vec: Vec<Vec<f64>> = Vec::new();
        for x in batch.iter(){
            let embedding = test_net.embed(x.clone());
            emb_vec.push(embedding.into_iter().flatten().collect());
        }
        let emb_array: Array2<f64> = Array2::from_shape_vec((batch.len(), 3*DIMENSIONS ), emb_vec.into_iter().flatten().collect()).unwrap();
        let hpreact: Array2<f64> = emb_array.clone().dot(&test_net.input_weights);
        let h = tanh(hpreact);
        let mut logits = h.dot(&test_net.output_weights) + &test_net.output_bias;
        subtract_row_max(&mut logits);
        let counts = logits.clone().mapv(|x| x.exp());
        let mut probs = counts.clone();
        for mut row in probs.axis_iter_mut(Axis(0)) {
            let row_sum: f64 = row.iter().sum::<f64>().clone();
            row /= row_sum;
        }

    assert!((probs.sum() - 2.0).abs() <= 0.0001); //all elements of our prob vector should add to 2, 
    //because there are 2 normalized rows in it (one for each context in input)
    }
    const HIDDEN_SIZE: usize = 200;

    #[test]
    fn test_backwards_pass() { //test shapes and values of gradients
        let test_net = Network::new();
        let batch_size = 2;
        let probs = Array2::from_shape_vec((batch_size, 27), vec![0.1; batch_size * 27]).unwrap();
        let h = Array2::from_shape_vec((batch_size, HIDDEN_SIZE), vec![0.5; batch_size * HIDDEN_SIZE]).unwrap();
        let emb_array = Array2::from_shape_vec((batch_size, BLOCK_SIZE * DIMENSIONS), vec![0.1; batch_size * BLOCK_SIZE * DIMENSIONS]).unwrap();
        let y_batch = vec![1, 2];
        let (grad_embedding, grad_input_weights, grad_output_weights, grad_output_bias) = test_net.backwards_pass(probs.clone(), h.clone(), emb_array.clone(), y_batch);
        //ensure shapes are correct
        assert_eq!(emb_array.shape(),grad_embedding.shape());
        assert_eq!(test_net.input_weights.shape(),grad_input_weights.shape());
        assert_eq!(test_net.output_weights.shape(),grad_output_weights.shape());
        assert_eq!(test_net.output_bias.shape(),grad_output_bias.shape());
        
        //ensure there are non zero values
        assert!(grad_embedding.iter().any(|&x| x != 0.0));
        assert!(grad_input_weights.iter().any(|&x| x != 0.0));
        assert!(grad_output_weights.iter().any(|&x| x != 0.0));
        assert!(grad_output_bias.iter().any(|&x| x != 0.0));
        
    }

}
