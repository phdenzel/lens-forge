use lens_forge::viz;


fn test_gradient_by_name(name: &str) {
    let width = 1300.00;
    let height = 70.00;
    let gradient = viz::get_gradient(name);
    let imgbuf = image::ImageBuffer::from_fn(width as u32, height as u32, |x, _| {
        image::Rgba(gradient.at(x as f32 / width).to_rgba8())
    });
    let filename: String = format!("gradient_{name}.png");
    match imgbuf.save(&filename) {
	Err(e) => eprintln!("Error writing image: {}", e),
	Ok(()) => println!("Image written: {}", filename),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box <dyn std::error::Error>> {
    println!("Hello, world!");
    test_gradient_by_name("magma");
    test_gradient_by_name("turbo");
    test_gradient_by_name("viridis");
    test_gradient_by_name("inferno");
    test_gradient_by_name("plasma");
    test_gradient_by_name("cividis");
    test_gradient_by_name("hertzsprung");
    test_gradient_by_name("obscura");
    test_gradient_by_name("gaseous");
    Ok(())
}
