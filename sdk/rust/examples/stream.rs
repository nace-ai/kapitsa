use kapitsa::Engine;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let options = args
        .next()
        .ok_or("provide engine-options JSON and request JSON")?;
    let request = args.next().ok_or("provide request JSON")?;
    let engine = Engine::new(&options)?;
    let mut request = engine.request(&request)?;
    let result = request.generate_with(|delta| {
        println!("{delta}");
        true
    })?;
    println!("{result}");
    Ok(())
}
