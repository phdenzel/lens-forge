use colorgrad::Gradient;
pub use crate::viz::gradients::{hertzsprung, obscura, gaseous};

pub fn get_gradient(name: &str) -> Box <dyn Gradient> {    
    match name.to_lowercase().as_str() {
        "viridis" => Box::new(colorgrad::preset::viridis()),
        "magma" => Box::new(colorgrad::preset::magma()),
        "inferno" => Box::new(colorgrad::preset::inferno()),
	"plasma" => Box::new(colorgrad::preset::plasma()),
	"turbo" => Box::new(colorgrad::preset::turbo()),
	"cividis" => Box::new(colorgrad::preset::cividis()),
	"hertzsprung" => Box::new(hertzsprung()),
	"obscura" => Box::new(obscura()),
	"gaseous" => Box::new(gaseous()),
        _ => Box::new(colorgrad::preset::viridis()),
    }
}
