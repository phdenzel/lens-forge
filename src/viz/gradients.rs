use colorgrad::Gradient;

pub fn hertzsprung() -> impl Gradient {
    colorgrad::GradientBuilder::new()
	.html_colors(&["#101016", "#6A1B36", "#852F4F", "#8F385B", "#D98647", "#EBC79A", "#FFFFFF"])
	.build::<colorgrad::CatmullRomGradient>()
	.unwrap()
}

pub fn obscura() -> impl Gradient {
    colorgrad::GradientBuilder::new()
	.html_colors(&["#101016", "#103F39", "#27524C", "#52746F", "#6F795D", "#D98647", "#EBC79A", "#FFFFFF"])
	.build::<colorgrad::CatmullRomGradient>()
	.unwrap()
}

pub fn gaseous() -> impl Gradient {
    colorgrad::GradientBuilder::new()
	.html_colors(&["#101016", "#1B366A", "#2F4F85", "#385B8F", "#79695d", "#D98647", "#EBC79A", "#FFFFFF"])
	.build::<colorgrad::CatmullRomGradient>()
	.unwrap()
}
